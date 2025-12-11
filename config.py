import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY not found in environment variables. Please add it to your .env file")

TMDB_BASE_URL = "https://api.themoviedb.org/3"
DATA_DIR = "data"
MODELS_DIR = "models"
MOVIES_CSV = f"{DATA_DIR}/movies.csv"
EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings.npy"
FAISS_INDEX_FILE = f"{DATA_DIR}/faiss_index.bin"