"""
Central config. Sab paths aur constants yahan rakhna, taaki har file me
hardcoded path na likhna pade.
"""
from pathlib import Path

# --- Project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Data paths ---
DATA_DIR = ROOT_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
STRUCTURED_DIR = DATA_DIR / "structured"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"

# tera master dataset
MASTER_CSV_PATH = STRUCTURED_DIR / "impp_master_v3.csv"

# SQL tool ka DB file (CSV ise load hoga)
DUCKDB_PATH = STRUCTURED_DIR / "macro.duckdb"
MACRO_TABLE_NAME = "macro_data"

# --- RAG tool paths ---
CHUNKS_JSONL_PATH = PROCESSED_DIR / "chunks.jsonl"
CHROMA_DB_DIR = PROCESSED_DIR / "chroma_db"

# --- Chunking config (baad me use hoga) ---
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 80

# --- Retrieval config (baad me use hoga) ---
TOP_K = 5

LLM_MODEL = "openai/gpt-oss-20b"
# openai/gpt-oss-20b is a reasoning model -- its internal "thinking" also
# consumes max_tokens. 600 was too tight for a question needing more
# reasoning (the model's budget ran out mid-thought, leaving zero tokens
# for the actual visible answer). 900 leaves enough headroom; combined
# with reasoning_effort="low" (set in orchestrator.py) to keep thinking
# itself short, this stays token-efficient overall.
MAX_TOKENS = 900

# --- SQL safety ---
SQL_ROW_LIMIT = 200  # agent kabhi is se zyada rows na maange
