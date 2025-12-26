"""
Configuration centralisée pour l'application CineSphere
Gère les chemins de fichiers et les paramètres de connexion à l'API TMDB
"""

import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY manquante dans les variables d'environnement")

TMDB_BASE_URL = "https://api.themoviedb.org/3"

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MOVIES_CSV = os.path.join(DATA_DIR, "raw", "movies.csv")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "processed", "embeddings_trained.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "processed", "faiss_index_trained.bin")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "processed", "training_pairs.csv")

FINE_TUNED_MODEL_PATH = os.path.join(MODELS_DIR, "fine_tuned", "movie_finder_v1")