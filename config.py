import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Dossiers
DATA_DIR = "data"
MODELS_DIR = "models"  # Nouveau dossier

# Fichiers de données
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")

# --- CHANGEMENT ICI ---
# Au lieu du nom en ligne, on met le chemin local
# MODEL_NAME = 'all-MiniLM-L6-v2'  <-- L'ancien
MODEL_PATH = os.path.join(MODELS_DIR, "film_bert") # <-- Le nouveau fine-tuné