import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your_key_here")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DOCS_PATH = os.path.join(BASE_DIR, "data", "docs")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

# Ensure directories exist
os.makedirs(DATA_DOCS_PATH, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
