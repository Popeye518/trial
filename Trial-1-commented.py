
# ============================================================
# CELL 1 — INSTALL DEPENDENCIES (commented out, run once)
# ============================================================
# Uncomment and run these once in your environment if not already installed.
# All required libraries for data processing, vector store, embeddings, AWS, and UI.

# !pip install pandas         # for reading Excel and data manipulation
# !pip install openpyxl       # pandas needs this to read .xlsx files
# !pip install chromadb       # vector database to store and search embeddings
# !pip install sentence-transformers  # for generating text embeddings + cross-encoder reranking
# !pip install boto3          # AWS SDK — used to call Amazon Bedrock (Claude LLM)
# !pip install gradio         # for building the interactive chat UI
# !pip install ipywidgets     # for rendering widgets inside Jupyter


# ============================================================
# CELL 2 — IMPORTS AND GLOBAL CONFIG
# ============================================================
# All imports are at the top so dependencies are clear at a glance.
# Config constants are centralized here — change them once and they propagate everywhere.

import pandas as pd
import re, json, uuid, csv, os
from datetime import datetime
from collections import Counter
import boto3
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── CONFIG ──────────────────────────────────────────────────────────────────
FILEPATH         = 'Updated_DataLake_Questionnaire.xlsx'   # Source Excel file with all 3 sheets
CHROMA_DIR       = './mbp_rag_db_advanced'                 # Local folder where ChromaDB persists embeddings
COLLECTION_NAME  = 'mbp_rag_collection_v2'                 # Name of the ChromaDB collection
EMBED_MODEL      = 'all-MiniLM-L6-v2'                      # Lightweight but accurate sentence embedding model
RERANK_MODEL     = 'cross-encoder/ms-marco-MiniLM-L-6-v2'  # Cross-encoder used to rerank retrieved chunks
BEDROCK_REGION   = 'ap-south-1'                            # AWS region where Bedrock is available
BEDROCK_MODEL_ID = 'anthropic.claude-3-haiku-20240307-v1:0' # Claude 3 Haiku — fast + accurate LLM on Bedrock
TOP_K_RETRIEVE   = 5    # How many chunks to pull from ChromaDB during semantic search
TOP_K_RERANK     = 3    # How many of those to keep after reranking (best 3 out of 5)
LOG_FILE         = 'chat_log.csv'  # Every Q&A gets logged here for audit/debugging

print('Config loaded.')


# ============================================================
# CELL 3 — SHARED UTILITY / HELPER FUNCTIONS
# ============================================================
# These small functions are used across all sheet loaders and chunkers.
# Defined early so everything below can call them freely.

def normalize_key(text: str) -> str:
    """
    Converts any string into a clean snake_case key.
    Used when building the global_governance_context dictionary.
    Example: "Project Name / ID" → "project_name_id"
    """
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9_]', '_', text)   # replace special chars with underscore
    return re.sub(r'_+', '_', text).strip('_') # collapse multiple underscores


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes all column names of a DataFrame to clean snake_case.
    This ensures consistent column access regardless of how the Excel was formatted.
    Example: "File Name (GIS/NDM)" → "file_name_gis_ndm"
    """
    def clean(col):
        col = str(col).strip().lower().replace(' ', '_')
        col = re.sub(r'[^a-z0-9_]', '_', col)
        return re.sub(r'_+', '_', col).strip('_')
    df.columns = [clean(c) for c in df.columns]
    return df


def build_standardized_file_key(filename: str):
    """
    Creates a stable, normalized identifier from a raw filename.
    This key is used to JOIN Sheet 2 (file info) with Sheet 3 (field mappings).

    Why we need this:
    - Raw filenames contain date stamps like YYYYMMDD or YYYY-MM-DD
    - These change per delivery but represent the same logical file
    - By stripping them, we get a stable key: "MBP_AccountExtract_4300_YYYYMMDD.txt" → "mbp_accountextract_4300"

    Steps:
    1. Lowercase and strip whitespace
    2. Remove .txt / .csv extensions
    3. Replace dashes with underscores (consistent separator)
    4. Strip 8-digit dates and literal YYYYMMDD/YYYY_MM_DD patterns
    5. Collapse duplicate underscores and strip trailing ones
    """
    if pd.isna(filename): return None
    fn = str(filename).lower().strip()
    fn = re.sub(r'\.txt|\.csv', '', fn)          # remove file extension
    fn = fn.replace('-', '_')                       # normalize separators
    fn = re.sub(r'\d{8}', '', fn)                  # remove 8-digit date stamps
    fn = re.sub(r'yyyy_mm_dd|yyyymmdd', '', fn)     # remove literal date placeholders
    return re.sub(r'_+', '_', fn).strip('_')        # clean up extra underscores

print('Helpers ready.')


# ============================================================
# CELL 4 — VERIFY EXCEL SHEET NAMES BEFORE LOADING
# ============================================================
# Reads the Excel file index and confirms all 3 expected sheets exist.
# This acts as a quick sanity check — if sheet names changed in the Excel,
# you'll get a clear error here instead of a confusing crash later.

import pandas as pd
xl = pd.ExcelFile(FILEPATH)
print('Sheets found in your Excel file:')
for s in xl.sheet_names:
    print(f'  → {s}')

# Define expected sheet names as constants — update here if Excel sheets are renamed
SHEET_QUESTIONNAIRE = 'Questionnaire'
SHEET_FILE_INFO     = 'File Information if File Based'
SHEET_MAPPING       = 'Metadata_OR_Mapping'

# Validate all 3 are present
missing = [s for s in [SHEET_QUESTIONNAIRE, SHEET_FILE_INFO, SHEET_MAPPING]
           if s not in xl.sheet_names]

if missing:
    print(f'\n❌ MISSING SHEETS: {missing}')
    print('Update SHEET_QUESTIONNAIRE / SHEET_FILE_INFO / SHEET_MAPPING above to match your actual sheet names.')
else:
    print('\n✅ All 3 sheets found. Safe to proceed.')


# ============================================================
# CELL 5 — SHEET LOADERS: READ AND CLEAN ALL 3 SHEETS
# ============================================================
# Each function handles one sheet. They all follow the same pattern:
#   1. Read raw Excel with no header assumption (header=None)
#   2. Drop fully empty rows
#   3. Dynamically detect the real header row by scanning for known keywords
#   4. Set that row as column names and slice data below it
#   5. Normalize column names to snake_case
#   6. Return the clean DataFrame

def load_questionnaire_sheet(filepath):
    """
    Loads Sheet 1 (Questionnaire) and splits it into two parts:
    - project_df     : the top 5 rows with project-level metadata (name, ID, impact score, etc.)
    - questionnaire_df : the main questionnaire responses (category, description, response)

    Why split?
    The top rows are used to build the global_governance_context dictionary (control layer).
    The bottom rows are chunked for semantic search in ChromaDB.
    """
    # Read raw without assuming any header row
    df = pd.read_excel(filepath, sheet_name='Questionnaire', header=None)
    df = df.dropna(how='all').reset_index(drop=True)

    # Lowercase all cells to safely detect the header row
    dflower = df.apply(lambda col: col.astype(str).str.strip().str.lower())

    # Find the row that contains "field", "description", "value" — that's our real header
    header_mask = dflower.apply(lambda r: {'field','description','value'}.issubset(set(r)), axis=1)
    if not header_mask.any():
        raise ValueError('Header row not found in Questionnaire')
    header_idx = header_mask.idxmax()

    # Set that row as column names, then take everything below it as data
    df.columns = df.iloc[header_idx].astype(str).str.strip().str.lower()
    df = df.iloc[header_idx+1:].reset_index(drop=True)

    # The questionnaire section starts where field == "category"
    # Everything above that line is project metadata; everything below is questionnaire data
    category_rows = df[df['field'].astype(str).str.strip().str.lower() == 'category']
    split_idx = category_rows.index[0]

    project_df        = df.iloc[:split_idx].copy().reset_index(drop=True)
    questionnaire_df  = df.iloc[split_idx+1:].copy().reset_index(drop=True)
    questionnaire_df.columns = ['category', 'description', 'response']

    # Strip whitespace from string cells
    questionnaire_df = questionnaire_df.apply(
        lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    return project_df, questionnaire_df


def load_file_information_sheet(filepath):
    """
    Loads Sheet 2 (File Information if File Based).
    Each row = one physical file delivered by the source system.
    Contains ingestion metadata: frequency, transmission type, schema, mailboxes, etc.

    Also adds a 'standardized_file_key' column — the stable join key used to
    link this sheet with Sheet 3 (field mappings).
    """
    raw = pd.read_excel(filepath, sheet_name='File Information if File Based', header=None)
    raw = raw.dropna(how='all').reset_index(drop=True)

    # Find the real header row by looking for the "file name" keyword
    dflower = raw.apply(lambda col: col.astype(str).str.strip().str.lower())
    header_mask = dflower.apply(lambda r: 'file name' in list(r), axis=1)
    if not header_mask.any():
        raise ValueError('Header not found in File Information')
    header_idx = header_mask.idxmax()

    df = raw.iloc[header_idx:].reset_index(drop=True)
    df.columns = df.iloc[0]           # first row becomes column headers
    df = df.iloc[1:].reset_index(drop=True)
    df = normalize_columns(df)        # snake_case all columns

    # Build the stable join key from the raw filename
    df['standardized_file_key'] = df['file_name'].apply(build_standardized_file_key)
    return df.reset_index(drop=True)


def load_metadata_mapping_sheet(filepath, file_information_df):
    """
    Loads Sheet 3 (Metadata_OR_Mapping).
    Each row = one field in a target table, linked to a specific file and record type.
    Contains: actual_file_name, record_type, schema_name, table_name, column_name, field_type, max_length, etc.

    Also adds 'standardized_file_key' so this sheet can be joined with Sheet 2.
    Note: No cross-validation with Sheet 2 here — that is done implicitly via file_tree construction.
    """
    raw = pd.read_excel(filepath, sheet_name='Metadata_OR_Mapping', header=None)
    raw = raw.dropna(how='all').reset_index(drop=True)

    # Find header row by scanning for "actual file name"
    dflower = raw.apply(lambda col: col.astype(str).str.strip().str.lower())
    header_mask = dflower.apply(lambda r: 'actual file name' in list(r), axis=1)
    if not header_mask.any():
        raise ValueError('Header not found in Metadata_OR_Mapping')
    header_idx = header_mask.idxmax()

    df = raw.iloc[header_idx:].reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df = normalize_columns(df)

    # Build the stable join key to link back to Sheet 2
    df['standardized_file_key'] = df['actual_file_name'].apply(build_standardized_file_key)
    return df.reset_index(drop=True)


# Load all three sheets
project_df, questionnaire_df = load_questionnaire_sheet(FILEPATH)
file_information_df           = load_file_information_sheet(FILEPATH)
metadata_mapping_df           = load_metadata_mapping_sheet(FILEPATH, file_information_df)

print(f'Sheet1 — project: {project_df.shape}, questionnaire: {questionnaire_df.shape}')
print(f'Sheet2 — file info: {file_information_df.shape}')
print(f'Sheet3 — mapping: {metadata_mapping_df.shape}')


# ============================================================
# CELL 6 — BUILD GLOBAL GOVERNANCE CONTEXT (CONTROL LAYER)
# ============================================================
# Extracts a flat key-value dictionary from both Sheet 1 sections.
# This is NOT chunked or embedded. It is used as:
#   - Metadata injected into every chunk (so the LLM always knows the project context)
#   - Direct lookup for deterministic governance questions (e.g., "who is the vendor?")
#   - Routing signal for domain detection

def extract_project_metadata(project_df):
    """
    Reads the top 5 rows of Sheet 1 (project-level fields like name, ID, impact score).
    Returns a dict like: { "project_name": "MBP 3.11 Upgrade", "impact_score": "90", ... }
    """
    metadata = {}
    for _, row in project_df.iterrows():
        key   = normalize_key(str(row['field']))
        value = row['value']
        if pd.notna(value):
            metadata[key] = str(value).strip()
    return metadata


def extract_governance_attributes(questionnaire_df):
    """
    Scans questionnaire rows for specific category/description signals and
    extracts structured governance attributes needed for routing and filtering.

    Why pattern-match instead of taking everything?
    We only want the critical control-layer fields here (source name, security class,
    environments, approval status, etc.). All other questionnaire content will be
    chunked and embedded separately for semantic search.
    """
    gov = {}
    for _, row in questionnaire_df.iterrows():
        cat  = str(row['category']).strip().lower()
        desc = str(row['description']).strip().lower()
        resp = row['response']
        if pd.isna(resp): continue
        resp = str(resp).strip()

        # Match known governance signals by category and description keywords
        if cat == 'source name':                                    gov['source_name']             = resp
        if cat == 'business line':                                  gov['business_line']            = resp
        if cat == 'data access'         and 'host type'       in desc: gov['host_type']            = resp
        if cat == 'data access'         and 'delivery method' in desc: gov['delivery_method']      = resp
        if cat == 'availability'        and 'how often'       in desc: gov['data_frequency']       = resp
        if cat == 'file db information' and 'security'        in desc: gov['security_classification'] = resp
        if cat == 'environments':                                   gov['environments']             = resp
        if cat == 'data retention period':                          gov['data_retention_period']    = resp
        if cat == 'source owner approval':                          gov['source_owner_approval']    = resp
    return gov


# Merge project metadata + governance attributes into one flat control dictionary
project_metadata      = extract_project_metadata(project_df)
governance_attributes = extract_governance_attributes(questionnaire_df)
global_governance_context = {**project_metadata, **governance_attributes}

print('Global Governance Context:')
for k, v in global_governance_context.items():
    print(f'  {k}: {v}')


# ============================================================
# CELL 7 — BUILD FILE TREE (IN-MEMORY HIERARCHY)
# ============================================================
# Merges Sheet 2 (file-level metadata) with Sheet 3 (field-level mappings)
# into a nested dictionary structure. This tree lives only in memory during chunking.
# It is NOT stored in ChromaDB. Its sole purpose is to organize data for chunk generation.
#
# Structure:
# file_tree = {
#   "mbp_accountextract_4300": {
#     "file_info": { ...row from Sheet 2... },
#     "mappings": {
#       "9000": {
#         "APD_AR_TDL_DP": [ {field row 1}, {field row 2}, ... ]
#       }
#     }
#   },
#   ...
# }

def build_file_tree(file_information_df, metadata_mapping_df):
    """
    Step 1: Initialize one node per file from Sheet 2 (13 files → 13 nodes)
    Step 2: For each row in Sheet 3, find the matching file node and nest the
            field under its record_type → table_name path.
    If a Sheet 3 row has a key that doesn't exist in Sheet 2, it is silently skipped.
    """
    file_tree = {}

    # Initialize file nodes from Sheet 2
    for _, row in file_information_df.iterrows():
        key = row['standardized_file_key']
        file_tree[key] = {'file_info': row.to_dict(), 'mappings': {}}

    # Populate field mappings from Sheet 3 under the matching file node
    for _, row in metadata_mapping_df.iterrows():
        key         = row['standardized_file_key']
        record_type = str(row.get('record_type', 'UNKNOWN'))
        table_name  = row.get('table_name', 'UNKNOWN')

        if key not in file_tree:
            continue  # Sheet 3 has a file not in Sheet 2 — skip gracefully

        # Create nested dict layers if they don't exist yet
        if record_type not in file_tree[key]['mappings']:
            file_tree[key]['mappings'][record_type] = {}
        if table_name not in file_tree[key]['mappings'][record_type]:
            file_tree[key]['mappings'][record_type][table_name] = []

        # Append this field row under file → record_type → table
        file_tree[key]['mappings'][record_type][table_name].append(row.to_dict())

    return file_tree


file_tree = build_file_tree(file_information_df, metadata_mapping_df)
print(f'File tree: {len(file_tree)} files')
print(f'Keys: {list(file_tree.keys())}')


# ============================================================
# CELL 8 — BUILD ALL CHUNKS (GOVERNANCE + FILE + TECHNICAL)
# ============================================================
# This is the core of the RAG data preparation.
# We create 3 types of chunks — each type goes into the same ChromaDB collection
# but is tagged with a 'domain' metadata field so the router can filter them.
#
# Chunk count breakdown:
#   - Governance chunks  : 1 project overview + 1 per questionnaire category = 26
#   - File info chunks   : 1 per physical file = 13
#   - Technical chunks   : 1 per (file × record_type × table) combination = 13
#   - TOTAL = 52 chunks


# ── GOVERNANCE CHUNKS (Sheet 1) ─────────────────────────────────────────────
def build_governance_chunks(project_df, questionnaire_df, global_governance_context):
    """
    Creates two types of governance chunks:
    1. One project_overview chunk — all key-value pairs from the top project metadata rows
    2. One chunk per questionnaire category — groups all Q&A rows under that category

    Each chunk gets 'domain': 'governance' in metadata so the router can filter to
    only governance chunks when answering questions about vendor, risk, retention, etc.
    """
    chunks = []

    # Chunk 1: Project overview — all project-level fields as "Field: Value" lines
    project_lines = [
        f"{row['field']}: {row['value']}"
        for _, row in project_df.iterrows() if pd.notna(row['value'])
    ]
    chunks.append({
        'content': '\n'.join(project_lines),
        'metadata': {
            'domain': 'governance', 'section': 'project_overview',
            'source_name':   global_governance_context.get('source_name', ''),
            'project_name':  global_governance_context.get('project_name', '')
        }
    })

    # Chunk 2..N: One chunk per questionnaire category
    for category, group in questionnaire_df.groupby('category'):
        lines = [
            f"{row['description']}: {row['response']}"
            for _, row in group.iterrows() if pd.notna(row['response'])
        ]
        if not lines: continue  # skip empty categories

        content = f"Category: {category}\n" + '\n'.join(lines)
        chunks.append({
            'content': content,
            'metadata': {
                'domain': 'governance', 'section': 'questionnaire',
                'category':     category.strip().lower(),
                'source_name':  global_governance_context.get('source_name', ''),
                'project_name': global_governance_context.get('project_name', '')
            }
        })
    return chunks


# ── FILE INFORMATION CHUNKS (Sheet 2) ───────────────────────────────────────
def build_file_information_chunks(file_information_df):
    """
    Creates one chunk per physical file from Sheet 2.
    Each chunk contains all the key ingestion details for that file:
    frequency, transmission type, load type, schema, mailboxes, encryption, etc.

    'None' values are filtered out so the chunk text stays clean.
    Metadata includes file_key, frequency, load_type, transmission, schema
    for ChromaDB filtering (e.g., "show me only Daily Delta files").
    """
    chunks = []
    for _, row in file_information_df.iterrows():
        lines = [
            f"File Name: {row.get('file_name')}",
            f"Description: {row.get('file_description_include_subject_area')}",
            f"Transmission Type: {row.get('transmission_type_gis_ndm_other')}",
            f"Frequency: {row.get('frequency_daily_weekly_monthly')}",
            f"Load Type: {row.get('full_delta')}",
            f"Autosys Start: {row.get('autosys_start')}",
            f"Schema: {row.get('datalake_schema_name_raw_cl')}",
            f"Encrypted: {row.get('encrypted_y_n')}",
            f"Compressed: {row.get('compressed_y_n')}",
            f"Producer Mailbox: {row.get('original_producer_mail_box')}",
            f"Consumer Mailbox: {row.get('consumer_mail_box')}",
            f"File Size: {row.get('file_size_bytes_')}",
            f"Holiday Rules: {row.get('holiday_rules')}",
            f"In Scope Acquisition: {row.get('in_scope_for_acquisition_y_n_if_n_then_reason')}",
            f"In Scope Ingestion: {row.get('in_scope_for_ingestion_y_n_if_n_then_reason')}",
        ]
        # Drop any lines that resolved to "None" (missing column values)
        content = '\n'.join([l for l in lines if 'None' not in str(l)])

        metadata = {
            'domain':       'file_information',
            'file_key':     str(row.get('standardized_file_key', '')),
            'frequency':    str(row.get('frequency_daily_weekly_monthly', '')),
            'load_type':    str(row.get('full_delta', '')),
            'transmission': str(row.get('transmission_type_gis_ndm_other', '')),
            'schema':       str(row.get('datalake_schema_name_raw_cl', ''))
        }
        chunks.append({'content': content, 'metadata': metadata})
    return chunks


# ── TECHNICAL MAPPING CHUNKS (Sheet 3) ──────────────────────────────────────
def build_technical_chunks(file_tree):
    """
    Creates one chunk per (file, record_type, table) combination from the file_tree.
    Each chunk includes:
    - File-level context (name, frequency, transmission, load type) from Sheet 2
    - Target schema and table name
    - A list of all fields in that table (field name, type, max length, description)

    This granularity means a user asking "what fields are in record type 9000
    of MBP Account Extract?" gets exactly the right chunk — no noise from other files.
    """
    chunks = []
    for file_key, file_data in file_tree.items():
        file_info = file_data['file_info']
        mappings  = file_data['mappings']

        for record_type, tables in mappings.items():
            for table_name, fields in tables.items():

                # Build human-readable header lines for this chunk
                lines = [
                    f"File: {file_info.get('file_name')}",
                    f"Record Type: {record_type}",
                    f"Target Schema: {fields[0].get('schema_name') if fields else ''}",
                    f"Target Table: {table_name}",
                    f"Frequency: {file_info.get('frequency_daily_weekly_monthly')}",
                    f"Transmission: {file_info.get('transmission_type_gis_ndm_other')}",
                    f"Load Type: {file_info.get('full_delta')}",
                    '\nFields:'
                ]

                # Append one line per field: "  - FIELD_NAME (TYPE, max=LEN): description"
                for f in fields:
                    lines.append(
                        f"  - {f.get('field_name')} ({f.get('field_type')}, "
                        f"max={f.get('max_length')}): "
                        f"{f.get('_field_description', f.get('field_description', ''))}"
                    )

                content = '\n'.join(lines)
                metadata = {
                    'domain':       'technical_mapping',
                    'file_key':     file_key,
                    'record_type':  str(record_type),
                    'schema':       str(fields[0].get('schema_name', '')) if fields else '',
                    'table':        str(table_name),
                    'frequency':    str(file_info.get('frequency_daily_weekly_monthly', '')),
                    'load_type':    str(file_info.get('full_delta', '')),
                    'transmission': str(file_info.get('transmission_type_gis_ndm_other', ''))
                }
                chunks.append({'content': content, 'metadata': metadata})
    return chunks


# Build all chunks from each domain and merge into one master list
governance_chunks  = build_governance_chunks(project_df, questionnaire_df, global_governance_context)
file_chunks        = build_file_information_chunks(file_information_df)
technical_chunks   = build_technical_chunks(file_tree)
all_chunks         = governance_chunks + file_chunks + technical_chunks

print(f'Governance: {len(governance_chunks)}, File: {len(file_chunks)}, Technical: {len(technical_chunks)}')
print(f'Total chunks: {len(all_chunks)}')


# ============================================================
# CELL 8.5 — ANALYTICAL FACTS LAYER (PRE-COMPUTED AGGREGATES)
# ============================================================
# RAG is great for semantic/descriptive questions but unreliable for counting.
# If a user asks "how many STRING fields are there?", semantic search might
# return a few example chunks but miss half the fields.
#
# Solution: compute all aggregate facts from the full DataFrames ONCE, store them
# in a Python dict, and inject them directly into the LLM prompt for count-type questions.
# This guarantees zero hallucination for any counting / listing / grouping query.

import pandas as pd

def build_analytical_facts(metadata_mapping_df, file_information_df):
    """
    Computes a comprehensive set of aggregate facts from both sheets.
    These facts are deterministic — computed from the full DataFrame, not from embeddings.

    Organized into three sections:
    A) Field-level analysis  (from Sheet 3 / Metadata_OR_Mapping)
    B) File scope & mailbox  (from Sheet 2 / File Information)
    C) Autosys timing        (from Sheet 2 / File Information)
    D) Cross-sheet join      (autosys hour → files + record types)
    """
    mdf = metadata_mapping_df.copy()
    fdf = file_information_df.copy()

    # Normalize column names to simple snake_case for safe access
    mdf.columns = [str(c).strip().lower().replace(' ', '_') for c in mdf.columns]
    fdf.columns = [str(c).strip().lower().replace(' ', '_') for c in fdf.columns]

    facts = {}

    # ── A) FIELD-LEVEL ANALYSIS (Sheet 3) ─────────────────────────────────
    # Dynamically find column names using keyword matching — handles minor naming variations
    fn_col = next((c for c in mdf.columns if c == 'field_name'           or 'field_name' in c), None)
    ft_col = next((c for c in mdf.columns if 'field' in c and 'type' in c), None)
    ml_col = next((c for c in mdf.columns if 'max'   in c and 'length' in c), None)
    rt_col = next((c for c in mdf.columns if 'record' in c and 'type'   in c), None)
    af_col = next((c for c in mdf.columns if 'actual' in c and 'file'   in c), None)

    # Fact 1: Total number of field rows in the mapping sheet
    facts['total_fields'] = len(mdf)

    # Fact 2: Field type distribution (e.g., STRING: 47, NUMBER: 17)
    # Also stores per-type field name lists for detailed lookups
    if ft_col:
        ft_series = mdf[ft_col]
        if isinstance(ft_series, pd.DataFrame): ft_series = ft_series.iloc[:, 0]  # handle duplicate col names
        ft_series = ft_series.astype(str).str.strip().str.upper()
        type_counts = ft_series.value_counts().to_dict()
        facts['field_type_distribution'] = type_counts
        for ftype, count in type_counts.items():
            key = ftype.lower()
            facts[f'total_{key}_fields'] = count
            if fn_col:
                facts[f'{key}_field_names'] = mdf.loc[ft_series == ftype, fn_col].astype(str).tolist()

    # Fact 3: Fields with max_length > 40 — often relevant for data quality / storage questions
    if ml_col and fn_col and ft_col:
        ml_series = mdf[ml_col]
        if isinstance(ml_series, pd.DataFrame): ml_series = ml_series.iloc[:, 0]
        mdf['_ml_num'] = pd.to_numeric(ml_series, errors='coerce')
        big_fields = mdf[mdf['_ml_num'] > 40][[fn_col, ft_col, ml_col]].dropna()
        facts['fields_with_max_length_gt_40']       = big_fields.to_dict(orient='records')
        facts['fields_with_max_length_gt_40_count'] = len(big_fields)
        facts['fields_with_max_length_gt_40_names'] = big_fields[fn_col].astype(str).tolist()

    # Fact 4: Unique record types and how many fields each has
    if rt_col:
        rt_series = mdf[rt_col]
        if isinstance(rt_series, pd.DataFrame): rt_series = rt_series.iloc[:, 0]
        rt_series = rt_series.astype(str).str.strip()
        facts['unique_record_types']       = sorted(rt_series.unique().tolist())
        facts['unique_record_types_count'] = len(facts['unique_record_types'])
        facts['fields_per_record_type']    = rt_series.value_counts().to_dict()

    # Fact 5: How many fields each file has (from actual_file_name column)
    if af_col:
        af_series = mdf[af_col]
        if isinstance(af_series, pd.DataFrame): af_series = af_series.iloc[:, 0]
        facts['fields_per_file'] = af_series.astype(str).str.strip().value_counts().to_dict()

    # ── B) FILE SCOPE & MAILBOX (Sheet 2) ─────────────────────────────────
    fn2_col  = next((c for c in fdf.columns if c == 'file_name'), None)
    acq_col  = next((c for c in fdf.columns if 'acquisition' in c), None)
    ing_col  = next((c for c in fdf.columns if 'ingestion'   in c), None)
    prod_col = next((c for c in fdf.columns if 'producer'    in c), None)
    cons_col = next((c for c in fdf.columns if 'consumer'    in c and 'mailbox' in c), None)
    auto_col = next((c for c in fdf.columns if 'autosys'     in c), None)

    # Fact 6: Files that are in scope for BOTH acquisition and ingestion
    if fn2_col and acq_col and ing_col:
        acq_series = fdf[acq_col].astype(str).str.strip().str.lower()
        ing_series = fdf[ing_col].astype(str).str.strip().str.lower()
        in_scope = fdf[acq_series.str.startswith('y') & ing_series.str.startswith('y')]
        facts['files_in_scope_acquisition_and_ingestion'] = in_scope[fn2_col].astype(str).tolist()
        facts['files_in_scope_count'] = len(in_scope)

    # Fact 7: Producer and consumer mailbox mappings per file + grouped by producer
    if fn2_col and prod_col:
        facts['producer_mailbox_per_file'] = dict(zip(fdf[fn2_col].astype(str), fdf[prod_col].astype(str)))
        facts['files_grouped_by_producer_mailbox'] = (
            fdf.groupby(prod_col)[fn2_col].apply(lambda s: s.astype(str).tolist()).to_dict()
        )
    if fn2_col and cons_col:
        facts['consumer_mailbox_per_file'] = dict(zip(fdf[fn2_col].astype(str), fdf[cons_col].astype(str)))

    # ── C) AUTOSYS TIMING (Sheet 2) ───────────────────────────────────────
    if fn2_col and auto_col:
        fdf['__autosys'] = fdf[auto_col].astype(str)

        # Raw autosys start time per file
        facts['autosys_start_per_file'] = dict(zip(fdf[fn2_col].astype(str), fdf['__autosys']))

        def parse_hour(val: str) -> int:
            """Extracts the numeric hour from strings like '3 AM EST', '10 AM EST'."""
            try:
                v = str(val).upper().replace('EST', '').strip()
                v = v.replace('AM', '').replace('PM', '').strip()
                return int(v.split(':')[0].strip())
            except Exception:
                return 99  # fallback for unparseable values

        fdf['__hour'] = fdf['__autosys'].apply(parse_hour)

        # Files that start before 6 AM — useful for early-morning pipeline monitoring
        before6 = fdf[fdf['__hour'] < 6]
        facts['files_starting_before_6am']       = dict(zip(before6[fn2_col].astype(str), before6['__autosys']))
        facts['files_starting_before_6am_count'] = len(before6)

        # Group files by their autosys start hour
        facts['files_per_autosys_hour'] = (
            fdf.groupby('__autosys')[fn2_col].apply(lambda s: s.astype(str).tolist()).to_dict()
        )

    # ── D) CROSS-SHEET JOIN: autosys hour → files + record types ──────────
    # Joins Sheet 2 (timing) with Sheet 3 (record types) to answer questions like:
    # "Which record types are processed at 5 AM?"
    if fn2_col and auto_col and af_col and rt_col:
        f_info = fdf[[fn2_col, '__autosys']].copy()
        f_info.columns = ['file_name_join', 'autosys_start']

        m_info = mdf[[af_col, rt_col]].drop_duplicates().copy()
        m_info.columns = ['file_name_join', 'record_type']

        f_info['file_name_join'] = f_info['file_name_join'].astype(str).str.strip()
        m_info['file_name_join'] = m_info['file_name_join'].astype(str).str.strip()

        merged = pd.merge(f_info, m_info, on='file_name_join', how='left')

        # For each autosys hour, list all files and their record types
        hour_rt = merged.groupby('autosys_start').apply(
            lambda x: {
                'files':        x['file_name_join'].astype(str).tolist(),
                'record_types': sorted(x['record_type'].astype(str).unique().tolist())
            }
        ).to_dict()
        facts['files_and_record_types_by_autosys_hour'] = hour_rt

    return facts


analytical_facts = build_analytical_facts(metadata_mapping_df, file_information_df)

print("=== Analytical Facts summary ===")
for k, v in analytical_facts.items():
    print(f"{k}: {v}")


# ============================================================
# CELL 10 — VALIDATE ALL CHUNKS BEFORE INSERTING INTO CHROMADB
# ============================================================
# A quick sanity check before we embed and insert.
# Catches any empty content or missing metadata fields early.

def validate_chunks(chunks):
    """
    Iterates all chunks and asserts:
    - content is not empty
    - metadata dict is present
    - 'domain' key exists in metadata (required for router-based filtering)

    Also prints a domain distribution count so you can verify the chunk split.
    """
    for i, chunk in enumerate(chunks):
        assert chunk.get('content'),             f'Chunk {i} has empty content'
        assert chunk.get('metadata'),            f'Chunk {i} missing metadata'
        assert 'domain' in chunk['metadata'],    f'Chunk {i} missing domain key'
    domain_dist = Counter(c['metadata']['domain'] for c in chunks)
    print(f'All {len(chunks)} chunks valid.')
    print(f'Domain distribution: {dict(domain_dist)}')

validate_chunks(all_chunks)


# ============================================================
# CELL 11 — INITIALIZE EMBEDDING MODEL AND CHROMADB
# ============================================================
# Load the sentence transformer model (runs locally, no API needed).
# Initialize a persistent ChromaDB client — data survives kernel restarts.
# Delete any existing collection to start fresh — avoids stale/duplicate vectors.

# Load embedding model
embedding_model = SentenceTransformer(EMBED_MODEL)
print(f'Embedding model loaded: {EMBED_MODEL}')

# Initialize ChromaDB with version-safe approach
# (API changed between chromadb < 0.4.x and >= 0.4.x)
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    print('Using PersistentClient (chromadb >= 0.4.x)')
except AttributeError:
    from chromadb.config import Settings
    chroma_client = chromadb.Client(Settings(
        persist_directory=CHROMA_DIR,
        is_persistent=True
    ))
    print('Using legacy Client (chromadb < 0.4.x)')

# Delete old collection if it exists (fresh insert every run to avoid duplicates)
try:
    chroma_client.delete_collection(COLLECTION_NAME)
    print('Deleted existing collection — fresh start.')
except:
    pass  # collection didn't exist yet, that's fine

collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
print(f'Collection ready: {COLLECTION_NAME}')


# ============================================================
# CELL 12 — EMBED ALL CHUNKS AND INSERT INTO CHROMADB
# ============================================================
# Converts all 52 chunk texts into numerical vectors using the embedding model,
# then stores them in ChromaDB along with their metadata and a unique UUID per chunk.
# Progress bar is shown because encoding 52 chunks takes a few seconds.

documents = [c['content']  for c in all_chunks]  # text content of each chunk
metadatas = [c['metadata'] for c in all_chunks]  # domain + filter metadata per chunk
ids       = [str(uuid.uuid4()) for _ in all_chunks]  # unique ID required by ChromaDB

print('Generating embeddings — please wait...')
embeddings = embedding_model.encode(
    documents, convert_to_numpy=True, show_progress_bar=True
).tolist()

# Insert everything into ChromaDB in one batch
collection.add(
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings,
    ids=ids
)
print(f'Inserted {len(documents)} chunks into ChromaDB.')


# ============================================================
# CELL 13 — QUERY ROUTER (DOMAIN + FILE KEY + RECORD TYPE DETECTION)
# ============================================================
# Before retrieving chunks, we classify the user's query to narrow the search space.
# This avoids returning irrelevant chunks (e.g., technical field details for a governance question).
#
# Three things are detected:
# 1. Domain      — governance / file_information / technical_mapping
# 2. File key    — which specific file is being asked about (if any)
# 3. Record type — a 4-5 digit number like 9000, 9200 (if mentioned)
#
# These signals are combined into a ChromaDB metadata filter for precision retrieval.

available_file_keys = list(file_tree.keys())

# Keywords that strongly suggest a governance-domain question
GOVERNANCE_KEYWORDS = [
    'vendor', 'risk', 'contact', 'retention', 'business line', 'security',
    'approval', 'availability', 'environment', 'sourcing', 'strategy',
    'masking', 'dr', 'disaster recovery', 'financial', 'audience',
    'party data', 'documentation', 'change management', 'quality issue',
    'products', 'subject area', 'ad group'
]
# Keywords that suggest a field/column/mapping question (technical domain)
TECHNICAL_KEYWORDS = [
    'field', 'column', 'record type', 'datatype', 'table', 'schema',
    'mapping', 'target table', 'max length', 'column name'
]
# Keywords that suggest a file-level ingestion question
FILE_KEYWORDS = [
    'frequency', 'transmission', 'encrypted', 'compressed',
    'autosys', 'load type', 'mailbox', 'file size', 'holiday',
    'delta', 'full', 'producer', 'consumer', 'in scope'
]


def detect_domain(query: str) -> tuple:
    """
    Scores the query against each keyword list and picks the domain with the highest hit count.
    Returns (domain_name, confidence) where confidence = winning_score / total_score.
    Falls back to 'governance' with 0.0 confidence when no keywords match.
    """
    q = query.lower()
    gov_score  = sum(1 for w in GOVERNANCE_KEYWORDS if w in q)
    tech_score = sum(1 for w in TECHNICAL_KEYWORDS  if w in q)
    file_score = sum(1 for w in FILE_KEYWORDS        if w in q)
    scores     = {'governance': gov_score, 'technical_mapping': tech_score, 'file_information': file_score}
    best_domain = max(scores, key=scores.get)
    best_score  = scores[best_domain]
    if best_score == 0:
        return 'governance', 0.0  # no signal found — default to governance
    confidence = best_score / (sum(scores.values()) + 1e-9)
    return best_domain, round(confidence, 2)


def detect_file_key(query: str) -> str:
    """
    Tries to identify which specific file the user is asking about.
    Cleans the query and checks how many significant parts of each file key appear in it.
    Requires a score >= 2 (at least 2 matching key segments > 3 chars) to avoid false positives.
    Returns the best matching file key or None if no confident match.
    """
    query_clean = re.sub(r'[^a-z0-9]', '', query.lower())
    best_match, best_score = None, 0
    for key in available_file_keys:
        key_parts = key.split('_')
        score = sum(2 for part in key_parts if len(part) > 3 and part in query_clean)
        if score > best_score:
            best_score = score
            best_match = key
    return best_match if best_score >= 2 else None


def detect_record_type(query: str) -> str:
    """
    Looks for a 4-5 digit number in the query (e.g., "9000", "11000").
    These are record type identifiers used in Sheet 3.
    Returns the matched string or None.
    """
    match = re.search(r'\b(\d{4,5})\b', query)
    return match.group() if match else None


def build_chroma_filter(domain=None, file_key=None, record_type=None):
    """
    Builds a ChromaDB metadata filter dict from the detected signals.
    - 0 conditions → None (no filter, full collection search)
    - 1 condition  → single condition dict
    - 2+ conditions → { "$and": [ cond1, cond2, ... ] }
    """
    conditions = []
    if domain:       conditions.append({'domain':      {'$eq': domain}})
    if file_key:     conditions.append({'file_key':    {'$eq': file_key}})
    if record_type:  conditions.append({'record_type': {'$eq': record_type}})
    if not conditions: return None
    if len(conditions) == 1: return conditions[0]
    return {'$and': conditions}

print('Query router ready.')


# ============================================================
# CELL 14 — RETRIEVAL + RERANKING PIPELINE
# ============================================================
# Two-stage retrieval:
# Stage 1 (Semantic search): Use embedding similarity to fetch top-5 candidate chunks from ChromaDB
# Stage 2 (Reranking):       Use a cross-encoder to re-score those 5 chunks and keep the best 3
#
# Why rerank?
# Embedding models are fast but approximate. The cross-encoder reads both the query AND
# each chunk together, making a more accurate relevance judgment. This improves answer quality.

# Load the cross-encoder reranker (may take a few seconds first time)
try:
    reranker     = CrossEncoder(RERANK_MODEL)
    USE_RERANKER = True
    print(f'Reranker loaded: {RERANK_MODEL}')
except Exception as e:
    reranker     = None
    USE_RERANKER = False
    print(f'Reranker unavailable ({e}), using semantic-only top-k.')


def retrieve_chunks(query: str, metadata_filter=None, top_k=TOP_K_RETRIEVE):
    """
    Encodes the query into a vector, then runs a similarity search in ChromaDB.
    If a metadata_filter is provided, only chunks matching that filter are considered.
    Returns (documents_list, metadatas_list).
    """
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()
    kwargs = {'query_embeddings': [query_embedding], 'n_results': top_k}
    if metadata_filter:
        kwargs['where'] = metadata_filter
    results = collection.query(**kwargs)
    return results['documents'][0], results['metadatas'][0]


def rerank_chunks(query, docs, metas, top_k=TOP_K_RERANK):
    """
    Re-scores retrieved chunks using the cross-encoder.
    Each (query, chunk) pair is scored; chunks are sorted by score descending.
    Falls back to plain top-k slice if reranker is unavailable or chunk count <= top_k.
    """
    if not USE_RERANKER or len(docs) <= top_k:
        return docs[:top_k], metas[:top_k]
    scores = reranker.predict([(query, doc) for doc in docs])
    ranked = sorted(zip(scores, docs, metas), reverse=True)
    return [d for _, d, _ in ranked[:top_k]], [m for _, _, m in ranked[:top_k]]


def route_and_retrieve(query: str):
    """
    Full routing + retrieval pipeline:
    1. Detect domain, file_key, record_type from query
    2. Build ChromaDB metadata filter
    3. Retrieve top-K chunks (with filter)
    4. Fallback to domain-only filter if < 2 results found (prevents empty context)
    5. Rerank and return final top-K chunks
    """
    domain, confidence = detect_domain(query)
    file_key           = detect_file_key(query)
    record_type        = detect_record_type(query)
    metadata_filter    = build_chroma_filter(domain=domain, file_key=file_key, record_type=record_type)

    docs, metas = retrieve_chunks(query, metadata_filter=metadata_filter, top_k=TOP_K_RETRIEVE)

    # Fallback: if too few results with strict filter, relax to domain-only
    if len(docs) < 2:
        fallback_filter = build_chroma_filter(domain=domain)
        docs, metas     = retrieve_chunks(query, metadata_filter=fallback_filter, top_k=TOP_K_RETRIEVE)

    docs, metas = rerank_chunks(query, docs, metas, top_k=TOP_K_RERANK)
    return docs, metas, domain, confidence, file_key, record_type

print('Retrieval + reranking pipeline ready.')


# ============================================================
# CELL 15 — BEDROCK LLM + ANSWER GENERATION WITH ANALYTICAL FACTS
# ============================================================
# Calls Amazon Bedrock (Claude 3 Haiku) to generate the final answer.
# Two types of context are injected into the prompt:
#   1. RAG context   — the top-3 reranked chunks (always)
#   2. Analytical facts — pre-computed aggregates (only for count/list questions)
#
# The system prompt enforces strict grounding — the LLM is not allowed to
# use any knowledge outside the provided context. This prevents hallucination.

import json
import boto3

# Connect to AWS Bedrock runtime service
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=BEDROCK_REGION
)

# System prompt: defines the LLM's role and strict rules
SYSTEM_PROMPT = """You are a Data Lake Governance and Technical Metadata Specialist
for the MBP 3.11 Upgrade project (FIS Profile source).

STRICT RULES:
1. Answer ONLY using the provided CONTEXT and ANALYTICAL FACTS. Never use outside knowledge.
2. Do NOT infer or fabricate any missing details.
3. If the answer is not in context, respond exactly:
   "The requested information is not available in the provided Data Lake documentation."
4. Be professional and structured. Use bullet points when listing multiple items.
5. For file questions: always include file name, frequency, and schema when available.
6. For field questions: always include field name, data type, and max length when available.
7. For COUNT / HOW MANY / TOTAL questions: always use the PRE-COMPUTED ANALYTICAL FACTS.
"""

# Pre-built facts summary string — injected into the prompt when the query is count/aggregate type
# This is formatted once and reused across all queries, keeping token usage efficient
facts_summary = f"""
PRE-COMPUTED ANALYTICAL FACTS (computed from full Excel, not from RAG):

FIELD TYPE / FIELDS:
- Total fields in mapping sheet : {analytical_facts.get('total_fields')}
- Field type distribution : {analytical_facts.get('field_type_distribution')}
- Total NUMBER fields : {analytical_facts.get('total_number_fields')}
- Total STRING fields : {analytical_facts.get('total_string_fields')}
- Fields with max length > 40 (count) : {analytical_facts.get('fields_with_max_length_gt_40_count')}
- Fields with max length > 40 (details) : {analytical_facts.get('fields_with_max_length_gt_40')}
- Unique record types : {analytical_facts.get('unique_record_types')}
- Unique record types count : {analytical_facts.get('unique_record_types_count')}
- Fields per record type : {analytical_facts.get('fields_per_record_type')}
- Fields per file : {analytical_facts.get('fields_per_file')}

FILE SCOPE / MAILBOX:
- Files in scope (acquisition+ingestion) : {analytical_facts.get('files_in_scope_count')}
- File names in scope : {analytical_facts.get('files_in_scope_acquisition_and_ingestion')}
- Producer mailbox per file : {analytical_facts.get('producer_mailbox_per_file')}
- Consumer mailbox per file : {analytical_facts.get('consumer_mailbox_per_file')}
- Files grouped by producer mailbox : {analytical_facts.get('files_grouped_by_producer_mailbox')}

TIMING / AUTOSYS:
- Autosys start per file : {analytical_facts.get('autosys_start_per_file')}
- Files starting before 6 AM EST : {analytical_facts.get('files_starting_before_6am')}
- Files starting before 6 AM count : {analytical_facts.get('files_starting_before_6am_count')}
- Files grouped by autosys start time : {analytical_facts.get('files_per_autosys_hour')}
- Files + record types by autosys time : {analytical_facts.get('files_and_record_types_by_autosys_hour')}
"""

# Keywords that indicate an aggregate/count/list question — trigger facts injection
AGG_KEYWORDS = [
    "how many", "count", "total", "number of",
    "how much", "list all", "all fields", "field type",
    "max length", "start before", "starting before", "start at", "autosys",
    "producer mailbox", "consumer mailbox"
]


def generate_answer(query: str, docs: list, conversation_history: list = None) -> str:
    """
    Builds the full prompt and sends it to Claude 3 Haiku via Amazon Bedrock.

    Prompt structure:
    [system prompt] + [analytical facts if needed] + [RAG context chunks] + [user question]

    Conversation history (last 4 turns) is prepended to support follow-up questions.
    Temperature is set to 0.0 for deterministic, factual answers.
    """
    # Join all retrieved chunks into one context block
    context = "\n---\n".join(map(str, docs))

    # Only inject analytical facts for count/aggregate/list queries
    q_lower    = query.lower()
    use_facts  = any(k in q_lower for k in AGG_KEYWORDS)
    extra_context = facts_summary if use_facts else ""

    # Build message list with conversation history (sliding window of last 4 turns)
    messages = []
    if conversation_history:
        for turn in conversation_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    # Assemble the full user prompt: system rules + optional facts + RAG context + question
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{extra_context}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"Answer:"
    )
    messages.append({"role": "user", "content": full_prompt})

    # Build Bedrock request payload (Claude Messages API format)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens":  800,
        "temperature": 0.0,   # zero temperature = deterministic, no creative variation
        "messages":    messages
    }

    # Call Bedrock and extract the text response
    resp = bedrock_runtime.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body)
    )
    return json.loads(resp["body"].read())["content"][0]["text"]


print("Bedrock client with analytical layer ready.")


# ============================================================
# CELL 16 — CONVERSATION LOGGER
# ============================================================
# Every Q&A interaction is appended to a CSV file for audit trail, debugging,
# and future analysis (e.g., most common questions, routing accuracy).
# Creates the file with headers on first run, then appends on subsequent calls.

def log_conversation(query, answer, domain, confidence, file_key, record_type):
    """
    Appends one row to the chat log CSV file.
    Fields logged: timestamp, query, answer, domain, confidence, file_key, record_type
    """
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'query', 'answer', 'domain', 'confidence', 'file_key', 'record_type'
        ])
        if not file_exists:
            writer.writeheader()  # write column names on first run
        writer.writerow({
            'timestamp':   datetime.now().isoformat(),
            'query':       query,
            'answer':      answer,
            'domain':      domain,
            'confidence':  confidence,
            'file_key':    file_key    or '',
            'record_type': record_type or ''
        })

print(f'Logger ready -> {LOG_FILE}')


# ============================================================
# CELL 17 — MAIN RAG PIPELINE FUNCTION
# ============================================================
# ask_rag_system() is the single entry point for all queries.
# It ties together: routing → retrieval → reranking → generation → memory → logging.
#
# Conversation memory is a sliding window of 20 messages (10 turns).
# When it exceeds 20, the oldest pair is dropped to stay within context limits.

conversation_history = []  # in-memory list of {"role": ..., "content": ...} dicts

def ask_rag_system(query: str, verbose: bool = True) -> str:
    """
    Full advanced RAG pipeline — executes these steps in order:
    1. Route   → detect domain, file_key, record_type from the query
    2. Retrieve → semantic search + metadata filter in ChromaDB (top-5)
    3. Rerank  → cross-encoder re-scores and selects best 3 chunks
    4. Generate → Claude 3 Haiku generates a grounded answer via Bedrock
    5. Memory  → question + answer appended to conversation_history
    6. Log     → row written to chat_log.csv
    """
    docs, metas, domain, confidence, file_key, record_type = route_and_retrieve(query)

    if verbose:
        print(f'[Router] domain={domain} | confidence={confidence} | '
              f'file_key={file_key} | record_type={record_type}')
        print(f'[Retrieval] {len(docs)} chunks after reranking')

    answer = generate_answer(query, docs, conversation_history)

    # Add this turn to memory (user message + assistant reply)
    conversation_history.append({'role': 'user',      'content': query})
    conversation_history.append({'role': 'assistant', 'content': answer})

    # Keep memory bounded — drop oldest turn when we exceed 20 messages (10 turns)
    if len(conversation_history) > 20:
        conversation_history.pop(0)
        conversation_history.pop(0)

    log_conversation(query, answer, domain, confidence, file_key, record_type)
    return answer


def reset_conversation():
    """Clears the in-memory conversation history. Call this to start a fresh chat session."""
    global conversation_history
    conversation_history = []
    print('Conversation history cleared.')

print('RAG pipeline ready. Use ask_rag_system("your question") to query.')


# ============================================================
# CELL 18 — END-TO-END TEST WITH SAMPLE QUESTIONS
# ============================================================
# Runs 10 test questions covering all three domains + one out-of-scope question.
# Expected behavior:
#   - Domain-specific questions → high confidence (1.0), correct answers
#   - Out-of-scope question ("CEO salary") → graceful "not available" response

test_questions = [
    'What is the frequency of MBP Account Extract?',      # file_information domain
    'Who is the vendor for this source?',                  # governance domain
    'What fields are in record type 9000 of MBP Account Extract?',  # technical_mapping domain
    'What is the data retention period?',                  # governance domain
    'Is MBP Account Extract encrypted?',                   # file_information domain
    'What are the environments available?',                # governance domain
    'What is the security classification of this source?', # governance domain
    'What is the autosys start time for the arrangement file?',  # file_information domain
    'What is the target table for MBP Transaction Extract?',     # technical_mapping domain
    'What is the CEO salary?',    # OUT-OF-SCOPE — should return "not available" response
]

reset_conversation()
for q in test_questions:
    print(f'\n{"="*70}')
    print(f'Q: {q}')
    answer = ask_rag_system(q)
    print(f'A: {answer}')
