"""
Module principal de recherche sémantique de films
Utilise FAISS pour la recherche vectorielle et un système de reranking hybride
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class MovieRetriever:
    """Système de recherche sémantique de films avec reranking hybride"""
    
    def __init__(self, model_path=None, use_trained=True):
        """
        Initialise le retriever avec un modèle pré-entraîné ou fine-tuné
        
        Args:
            model_path: Chemin personnalisé vers le modèle
            use_trained: Si True, utilise le modèle fine-tuné
        """
        if model_path is None and use_trained:
            model_path = config.FINE_TUNED_MODEL_PATH
        
        if model_path and not os.path.exists(model_path):
            print(f"Modèle fine-tuné introuvable à {model_path}")
            print(f"Utilisation du modèle de base")
            model_path = 'all-MiniLM-L6-v2'
        elif model_path is None:
            model_path = 'all-MiniLM-L6-v2'
        
        if 'trained' in str(model_path) or 'fine_tuned' in str(model_path):
            print(f"Chargement du modèle FINE-TUNÉ: {model_path}")
        else:
            print(f"Chargement du modèle de base: {model_path}")
        
        self.model = SentenceTransformer(model_path)
        self.movies_df = None
        self.index = None
        self.embeddings = None
        
    def load_movies(self, csv_path):
        """
        Charge le dataset de films depuis CSV
        
        Args:
            csv_path: Chemin vers movies.csv
        """
        print(f"Chargement des films depuis {csv_path}")
        self.movies_df = pd.read_csv(csv_path)
        print(f"{len(self.movies_df)} films chargés")
        return self.movies_df
    
    def create_movie_text(self, row):
        """
        Crée une représentation textuelle enrichie d'un film
        Doit correspondre exactement au format utilisé pendant l'entraînement
        
        Args:
            row: Ligne pandas d'un film
        
        Returns:
            Chaîne de texte combinant titre, genres, keywords et plot
        """
        text_parts = []
        
        if pd.notna(row['title']):
            text_parts.append(f"{row['title']} {row['title']}")
        
        if pd.notna(row['genres']):
            genres = row['genres']
            text_parts.append(f"{genres} {genres} {genres}")
        
        if pd.notna(row['keywords']):
            keywords = row['keywords']
            text_parts.append(f"{keywords} {keywords} {keywords} {keywords}")
        
        if pd.notna(row['plot']) and len(str(row['plot'])) > 10:
            plot = str(row['plot'])[:400]
            text_parts.append(plot)
        
        if pd.notna(row.get('year')) and pd.notna(row.get('rating')):
            text_parts.append(f"{row['year']} film rated {row['rating']}")
        
        return " ".join(text_parts)
    
    def generate_embeddings(self):
        """
        Génère les embeddings pour tous les films du dataset
        
        Returns:
            Matrice numpy des embeddings (n_movies, embedding_dim)
        """
        print("Génération des embeddings pour tous les films...")
        
        movie_texts = self.movies_df.apply(self.create_movie_text, axis=1).tolist()
        embeddings = self.model.encode(movie_texts, show_progress_bar=True, batch_size=64)
        
        print(f"Embeddings générés: {embeddings.shape}")
        return embeddings
    
    def build_index(self, embeddings):
        """
        Construit l'index FAISS pour la recherche rapide
        
        Args:
            embeddings: Matrice des embeddings
        """
        print("Construction de l'index FAISS...")
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index construit avec {self.index.ntotal} vecteurs")
        
    def save_index(self, index_path, embeddings_path):
        """Sauvegarde l'index FAISS et les embeddings"""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        
        print(f"Sauvegarde de l'index FAISS: {index_path}")
        faiss.write_index(self.index, index_path)
        
        print(f"Sauvegarde des embeddings: {embeddings_path}")
        np.save(embeddings_path, self.embeddings)
        
    def load_index(self, index_path, embeddings_path):
        """Charge l'index FAISS et les embeddings depuis le disque"""
        print(f"Chargement de l'index FAISS: {index_path}")
        self.index = faiss.read_index(index_path)
        
        print(f"Chargement des embeddings: {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
    
    def search(self, query, top_k=5, boost_rating=True, min_score=0.45, adaptive=True):
        """
        Recherche sémantique avec reranking hybride
        
        Args:
            query: Requête en langage naturel
            top_k: Nombre de résultats à retourner
            boost_rating: Active le reranking par rating et popularité
            min_score: Score minimum de pertinence
            adaptive: Filtre adaptatif des résultats
        
        Returns:
            Liste de dictionnaires avec les films les plus pertinents
        """
        query_embedding = self.model.encode([query])
        
        search_k = top_k * 4 if boost_rating else top_k
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            movie = self.movies_df.iloc[idx]
            similarity_score = 1 / (1 + distance)
            
            if boost_rating:
                rating_normalized = movie['rating'] / 10.0
                rating_weight = min(rating_normalized * 1.2, 0.95)
                popularity_normalized = min(movie['popularity'] / 50.0, 1.0)
                
                is_documentary = 'Documentary' in str(movie['genres'])
                doc_penalty = 0.85 if is_documentary else 1.0
                
                final_score = (similarity_score * 0.65 + 
                              rating_weight * 0.25 + 
                              popularity_normalized * 0.10) * doc_penalty
            else:
                final_score = similarity_score
            
            results.append({
                'title': movie['title'],
                'year': movie['year'],
                'genres': movie['genres'],
                'plot': movie['plot'],
                'keywords': movie['keywords'],
                'rating': movie['rating'],
                'popularity': movie['popularity'],
                'similarity_score': similarity_score,
                'final_score': final_score,
                'poster_path': movie.get('poster_path', '')
            })
        
        results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        if adaptive:
            filtered_results = []
            for r in results:
                if r['final_score'] >= min_score:
                    filtered_results.append(r)
                    if r['final_score'] < 0.55 and len(filtered_results) >= 3:
                        break
                    elif len(filtered_results) >= top_k:
                        break
            
            if len(filtered_results) < 3 and len(results) >= 3:
                return results[:3]
            
            return filtered_results if filtered_results else results[:top_k]
        
        return results[:top_k]


def main():
    """Reconstruit l'index FAISS avec le modèle fine-tuné"""
    print("\n" + "="*70)
    print("RECONSTRUCTION DE L'INDEX FAISS AVEC LE MODÈLE FINE-TUNÉ")
    print("="*70 + "\n")
    
    retriever = MovieRetriever(use_trained=True)
    retriever.load_movies(config.MOVIES_CSV)
    
    print("\nGénération des embeddings avec le modèle entraîné...")
    embeddings = retriever.generate_embeddings()
    retriever.embeddings = embeddings
    
    retriever.build_index(embeddings)
    retriever.save_index(config.FAISS_INDEX_FILE, config.EMBEDDINGS_FILE)
    
    print("\n" + "="*70)
    print("Test du retriever avec des requêtes exemples:")
    print("="*70)
    
    test_queries = [
        "romantic movie on a cruise ship",
        "toys that come to life",
        "AI falls in love with lonely writer"
    ]
    
    for query in test_queries:
        print(f"\nRequête: '{query}'")
        print("-" * 50)
        results = retriever.search(query, top_k=3)
        
        for i, movie in enumerate(results, 1):
            print(f"{i}. {movie['title']} ({movie['year']}) - Score: {movie['final_score']:.3f}")
    
    print("\n" + "="*70)
    print("Index prêt pour la production")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()