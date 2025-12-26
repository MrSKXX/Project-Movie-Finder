# Data Directory

## Structure
```
data/
├── raw/
│   └── movies.csv          # Dataset TMDB (télécharger avec data_fetcher.py)
└── processed/
    ├── training_pairs_train.csv
    ├── training_pairs_val.csv
    ├── embeddings_trained.npy    # Généré automatiquement
    └── faiss_index_trained.bin   # Généré automatiquement
```

## Fichiers Non Versionnés

Les fichiers suivants sont générés automatiquement et ne sont PAS dans Git:
- `embeddings*.npy` - Embeddings des films (généré par movie_retriever.py)
- `faiss_index*.bin` - Index FAISS (généré par movie_retriever.py)

Pour les générer:
```bash
python -m src.movie_retriever
```

## Fichiers Partagés

Le dataset `raw/movies.csv` peut optionnellement être partagé via:
1. Git LFS (Large File Storage)
2. Google Drive
3. Demander aux collègues de le générer: `python -m src.data_fetcher`