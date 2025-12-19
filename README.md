# 🎬 CineSphere - AI Movie Finder

CineSphere est un moteur de recherche de films sémantique de nouvelle génération. Contrairement aux recherches classiques par mots-clés, il utilise l'intelligence artificielle (Sentence Transformers) pour comprendre le **sens** et le **contexte** de la requête utilisateur.

## 🚀 Fonctionnalités Clés

* **Recherche Sémantique :** Comprend des requêtes complexes comme *"un film triste dans l'espace"* ou *"romance on a sinking boat"*.
* **Algorithme de Ranking Hybride :** Combine trois facteurs pour la pertinence :
    1.  **Similarité Sémantique (65%)** : Basée sur les embeddings (Vector Search).
    2.  **Qualité du Film (25%)** : Basée sur la note critique.
    3.  **Popularité (10%)** : Basée sur la tendance actuelle.
* **Query Expansion :** Enrichissement automatique de la requête utilisateur pour élargir le champ de recherche.
* **Interface Moderne :** Frontend React fluide avec un design immersif.

## 🛠️ Stack Technique

* **Backend :** Flask (Python)
* **AI/NLP :** `sentence-transformers` (Modèle `all-MiniLM-L6-v2`), `faiss-cpu` (Indexation vectorielle rapide).
* **Data :** Dataset TMDB (The Movie Database).
* **Frontend :** React.js (Single Page Application).

## 📦 Installation et Lancement

### 1. Pré-requis
* Python 3.8+
* Un environnement virtuel est recommandé.

### 2. Installation
```bash
# Cloner le projet (si via git) ou extraire le dossier
cd AiMovieFinder

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows :
venv\Scripts\activate
# Mac/Linux :
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt