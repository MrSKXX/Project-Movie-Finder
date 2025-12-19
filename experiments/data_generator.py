import pandas as pd
import os
import sys

# Import config du dossier parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class TrainingDataGenerator:
    def __init__(self):
        self.movies_df = None
        
    def load_movies(self, csv_path):
        print(f"Loading movies from {csv_path}")
        self.movies_df = pd.read_csv(csv_path)
        
    def create_movie_text(self, row):
        text_parts = []
        if pd.notna(row['title']): text_parts.append(str(row['title']))
        if pd.notna(row['genres']): text_parts.append(str(row['genres']))
        if pd.notna(row['keywords']): text_parts.append(str(row['keywords']))
        if pd.notna(row['plot']) and len(str(row['plot'])) > 10:
            plot = str(row['plot'])
            text_parts.append(plot[:300] if len(plot) > 300 else plot)
        return " ".join(text_parts)
    
    def generate_genre_pairs(self):
        # On crée des paires positives intelligentes
        genre_templates = {
            'Action': ['action movie', 'high octane action film', 'fights and explosions'],
            'Comedy': ['funny movie', 'hilarious comedy', 'make me laugh'],
            'Horror': ['scary movie', 'horror film', 'terrifying story'],
            'Romance': ['love story', 'romantic movie', 'falling in love'],
            'Science Fiction': ['sci-fi movie', 'space and technology', 'future world'],
            'Thriller': ['suspenseful movie', 'thriller', 'mystery and tension'],
            'Drama': ['drama movie', 'emotional story', 'serious film'],
            'Fantasy': ['fantasy world', 'magic and mythical creatures', 'fantasy adventure'],
            'Animation': ['animated movie', 'cartoon', 'animation film']
        }
        
        pairs = []
        for _, row in self.movies_df.iterrows():
            movie_text = self.create_movie_text(row)
            if pd.notna(row['genres']):
                for genre in row['genres'].split(', '):
                    if genre in genre_templates:
                        for query in genre_templates[genre]:
                            # Format pour MNRL : [Query, Positive Doc] (Pas de label)
                            pairs.append({'query': query, 'movie_text': movie_text})
        return pairs

    def generate_keyword_pairs(self):
        pairs = []
        for _, row in self.movies_df.iterrows():
            movie_text = self.create_movie_text(row)
            if pd.notna(row['keywords']):
                keywords = row['keywords'].split(', ')
                import random
                selected_keywords = random.sample(keywords, min(len(keywords), 3))
                
                for kw in selected_keywords:
                    query = f"movie about {kw}"
                    pairs.append({'query': query, 'movie_text': movie_text})
        return pairs

    def save_training_data(self, output_path):
        pairs = self.generate_genre_pairs() + self.generate_keyword_pairs()
        
        df = pd.DataFrame(pairs)
        # Mélanger les données est CRUCIAL pour MNRL
        df = df.sample(frac=1).reset_index(drop=True)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} positive pairs to {output_path}")

if __name__ == "__main__":
    gen = TrainingDataGenerator()
    gen.load_movies(config.MOVIES_CSV)
    gen.save_training_data(config.TRAINING_DATA_PATH)