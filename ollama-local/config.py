CSV_PATH = "../data-cleaning/processed_data/processed_data.csv"
OUTPUT_PATH = "output/output.csv"
KNOWLEDGE_DIR = "knowledge"

MODEL = "gemma3:12b"
EMBED_MODEL = "embeddinggemma"

TEXT_COLUMN = "analysis_text"

LIMIT = 100
MAX_CHARS = 800
TIMEOUT = 120

TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
