import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY', 'fb9fd64171c68091b1a6eac636c7e957')
TMDB_BASE_URL = "https://api.themoviedb.org/3"
DATA_DIR = "data"
MODELS_DIR = "models"
MOVIES_CSV = f"{DATA_DIR}/movies.csv"
EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings.npy"
FAISS_INDEX_FILE = f"{DATA_DIR}/faiss_index.bin"