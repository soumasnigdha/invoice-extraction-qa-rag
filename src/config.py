import os
from pathlib import Path

class Config:
    # Ollama settings
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Embedding settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # Vector Database settings
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "output/vector_db/chroma")
    COLLECTION_NAME = "financial_documents"
    
    # RAG settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SIMILARITY_THRESHOLD = 0.7
    MAX_RELEVANT_CHUNKS = 5
    
    # Paths - All relative to project root
    BASE_DIR = Path(__file__).parent.parent  # Go up from src/
    SRC_DIR = BASE_DIR / "src"
    
    # Input and Output directories at root level
    INPUT_DIR = BASE_DIR / "input"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Specific input directories
    INVOICES_DIR = INPUT_DIR / "invoices"
    BANK_STATEMENTS_DIR = INPUT_DIR / "bank_statements"
    RECEIPTS_DIR = INPUT_DIR / "receipts"
    
    # Specific output directories
    INDIVIDUAL_INVOICES_DIR = OUTPUT_DIR / "individual_invoices"
    MASTER_DATA_DIR = OUTPUT_DIR / "master_data"
    VECTOR_DB_DIR = OUTPUT_DIR / "vector_db"
    LOGS_DIR = OUTPUT_DIR / "logs"
    
    # Template and prompt directories in src
    TEMPLATES_DIR = SRC_DIR / "templates"
    PROMPTS_DIR = SRC_DIR / "prompts"
    
    # Excel settings
    MASTER_EXCEL_FILE = "master_financial_data.xlsx"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "output/logs/app.log")
    
    # Create directories
    for dir_path in [
        INPUT_DIR, INVOICES_DIR, BANK_STATEMENTS_DIR, RECEIPTS_DIR,
        OUTPUT_DIR, INDIVIDUAL_INVOICES_DIR, MASTER_DATA_DIR, 
        VECTOR_DB_DIR, LOGS_DIR
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
