# CineSphere - Moteur de Recherche Sémantique de Films

CineSphere est un système de recherche de films utilisant l'intelligence artificielle pour comprendre le sens des requêtes plutôt que de simples mots-clés.

## Caractéristiques

- **Recherche Sémantique**: Comprend des requêtes complexes comme "film triste dans l'espace" ou "romance sur un bateau qui coule"
- **Fine-Tuning avec MNRL**: Modèle all-MiniLM-L6-v2 fine-tuné avec MultipleNegativesRankingLoss
- **Curriculum Learning**: Données d'entraînement organisées en 4 niveaux de difficulté
- **Reranking Hybride**: Combine similarité sémantique (65%), qualité (25%) et popularité (10%)
- **Interface Web Moderne**: Frontend React avec design immersif

## Architecture
```
Requête utilisateur
    ↓
all-MiniLM-L6-v2 (fine-tuné)
    ↓
Index FAISS (recherche vectorielle)
    ↓
Reranking hybride
    ↓
Top-K résultats
```

## Stack Technique

- **Backend**: Flask (Python)
- **ML/NLP**: sentence-transformers, faiss-cpu
- **Data**: TMDB API (3,230 films)
- **Frontend**: React.js

## Installation

### Prérequis
- Python 3.8+
- Environnement virtuel recommandé

### Étapes
```bash
# Cloner le projet
git clone <repository-url>
cd AiMovieFinder

# Créer environnement virtuel
python -m venv venv

# Activer environnement
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Installer dépendances
pip install -r requirements.txt

# Configurer clé API TMDB
cp .env.example .env
# Éditer .env et ajouter votre TMDB_API_KEY
```

## Utilisation

### 1. Récupérer les données
```bash
python -m src.data_fetcher
```

### 2. Générer les données d'entraînement
```bash
python -m training.data_generator
```

### 3. Entraîner le modèle
```bash
python -m training.train
```

### 4. Évaluer le modèle
```bash
python -m training.evaluate
```

### 5. Construire l'index FAISS
```bash
python -m src.movie_retriever
```

### 6. Lancer l'application

**Option A: Interface Web**
```bash
python -m src.app
# Ouvrir frontend/index.html dans le navigateur
```

**Option B: Interface CLI**
```bash
python main.py
```

## Structure du Projet
```
AiMovieFinder/
├── src/                    # Code source principal
│   ├── config.py          # Configuration
│   ├── data_fetcher.py    # Récupération données TMDB
│   ├── movie_retriever.py # Moteur de recherche
│   └── app.py             # API Flask
├── training/              # Pipeline d'entraînement
│   ├── data_generator.py  # Génération données
│   ├── train.py           # Entraînement
│   └── evaluate.py        # Évaluation
├── data/                  # Données
│   ├── raw/              # Données brutes
│   └── processed/        # Données traitées
├── models/               # Modèles entraînés
│   └── fine_tuned/
├── frontend/             # Interface utilisateur
│   └── index.html
├── docs/                 # Documentation
└── main.py              # Point d'entrée CLI
```

## Résultats

### Métriques de Performance

| Métrique | Base | Fine-tuné | Amélioration |
|----------|------|-----------|--------------|
| MRR | 0.407 | 0.611 | +50.1% |
| Precision@1 | 30.0% | 50.0% | +66.7% |
| Precision@5 | 55.0% | 70.0% | +27.3% |
| Recall@10 | 60.0% | 85.0% | +41.7% |

### Exemples de Requêtes
```
"romantic movie on a sinking cruise ship"
→ Titanic ✓

"AI falls in love with lonely writer"
→ Her ✓

"dreams within dreams heist"
→ Inception ✓

"chef rat controls human cooking"
→ Ratatouille ✓
```

## Méthodologie

### 1. Génération des Données

**Curriculum Learning en 4 niveaux:**
- Niveau 1: Requêtes de genre (24,780 paires)
- Niveau 2: Requêtes thématiques (394 paires)
- Niveau 3: Requêtes basées sur le plot (1,688 paires)
- Niveau 4: Requêtes multi-concepts (2,350 paires)

### 2. Entraînement

- **Loss**: MultipleNegativesRankingLoss
- **Modèle de base**: all-MiniLM-L6-v2
- **Époques**: 3
- **Batch size**: 32
- **Learning rate**: 2e-5
- **Temps d'entraînement**: ~6 minutes (CPU)

### 3. Évaluation

- 20 requêtes de test avec ground truth
- Métriques: MRR, Precision@K, Recall@K

## API REST

### POST /api/search

Recherche sémantique de films.

**Request:**
```json
{
  "query": "romantic movie on a cruise ship",
  "top_k": 10
}
```

**Response:**
```json
{
  "results": [
    {
      "title": "Titanic",
      "year": "1997",
      "genres": "Drama, Romance",
      "rating": 7.9,
      "final_score": 0.812,
      "plot": "...",
      "poster_path": "/..."
    }
  ]
}
```

### GET /api/health

Vérification de l'état du serveur.

**Response:**
```json
{
  "status": "ok",
  "movies_loaded": 3230,
  "model_type": "fine-tuned"
}
```

## Développement

### Tests
```bash
python -m pytest tests/
```

### Linting
```bash
python -m flake8 src/ training/
```

## Licence

MIT License

## Auteurs

AiMovieFinder Team - Projet Universitaire 2025

## Remerciements

- TMDB pour l'API de données
- Sentence-Transformers pour le framework
- FAISS pour la recherche vectorielle