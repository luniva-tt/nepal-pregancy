import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "vectorstore")
REPORT_DIR = os.path.join(BASE_DIR, "pdf_reports")

# Model Configuration
LLM_MODEL = "mistral" # or "mistral"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 3

# Risk Thresholds
RISK_THRESHOLDS = {
    "bp_systolic_high": 140,
    "bp_diastolic_high": 90,
    "glucose_high": 140, # mg/dL random
    "heart_rate_high": 100,
    "heart_rate_low": 60
}

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
