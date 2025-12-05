import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
import config

class MovieRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.movies_df = None
        self.index = None
        
    def load_movies(self, csv_path):
        print(f"Loading movies from {csv_path}")
        self.movies_df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.movies_df)} movies")
        return self.movies_df
    
    def create_movie_text(self, row):
        text_parts = []
        
        if pd.notna(row['title']):
            text_parts.append(row['title'])
        
        if pd.notna(row['genres']):
            genres = row['genres']
            text_parts.append(f"{genres} {genres}")
        
        if pd.notna(row['keywords']):
            keywords = row['keywords']
            text_parts.append(f"{keywords} {keywords} {keywords} {keywords} {keywords}")
        
        if pd.notna(row['plot']) and len(str(row['plot'])) > 10:
            plot = str(row['plot'])
            if len(plot) > 200:
                plot = plot[:200]
            text_parts.append(plot)
        
        return " ".join(text_parts)
    
    def generate_embeddings(self):
        print("Generating embeddings for all movies...")
        
        movie_texts = self.movies_df.apply(self.create_movie_text, axis=1).tolist()
        embeddings = self.model.encode(movie_texts, show_progress_bar=True)
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def build_index(self, embeddings):
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors")
        
    def save_index(self, index_path, embeddings_path):
        print(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, index_path)
        
        print(f"Saving embeddings to {embeddings_path}")
        np.save(embeddings_path, self.embeddings)
        
    def load_index(self, index_path, embeddings_path):
        print(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading embeddings from {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
        
    def expand_query(self, query):
        synonyms = {
            'cruise ship': 'cruise ship boat vessel ocean liner ship sailing',
            'spaceship': 'spaceship spacecraft space ship vessel',
            'haunted house': 'haunted house mansion manor ghost',
            'time travel': 'time travel time machine temporal',
            'superhero': 'superhero hero comic book marvel dc',
            'zombie': 'zombie undead walking dead',
            'vampire': 'vampire dracula bloodsucker',
            'detective': 'detective investigator mystery crime',
            'heist': 'heist robbery theft stealing',
            'war': 'war battle combat military',
        }
        
        expanded = query.lower()
        for term, expansion in synonyms.items():
            if term in expanded:
                expanded = expanded.replace(term, expansion)
        
        return expanded
    
    def search(self, query, top_k=5, boost_rating=True, min_score=0.45, adaptive=True):
        expanded_query = self.expand_query(query)
        query_embedding = self.model.encode([expanded_query])
        
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
    retriever = MovieRetriever()
    
    retriever.load_movies(config.MOVIES_CSV)
    
    embeddings = retriever.generate_embeddings()
    retriever.embeddings = embeddings
    
    retriever.build_index(embeddings)
    
    os.makedirs(config.DATA_DIR, exist_ok=True)
    retriever.save_index(config.FAISS_INDEX_FILE, config.EMBEDDINGS_FILE)
    
    print("\n" + "="*50)
    print("Testing the retriever with sample queries:")
    print("="*50)
    
    test_queries = [
        "romance on a cruise ship",
        "action movie with aliens in space",
        "horror movie in a haunted house"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        results = retriever.search(query, top_k=3)
        
        for i, movie in enumerate(results, 1):
            print(f"{i}. {movie['title']} ({movie['year']}) - {movie['genres']}")
            print(f"   Score: {movie['similarity_score']:.3f}")

if __name__ == "__main__":
    main()