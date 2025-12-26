# experiments/advanced_data_generator.py

import pandas as pd
import random
from typing import List, Dict
import re
import os
import sys

# Fix the path - go up one directory from experiments/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import config

class AdvancedTrainingDataGenerator:
    def __init__(self, movies_df):
        self.movies_df = movies_df
        
    def extract_entities(self, text):
        """Extract names, places, concepts from plot"""
        if not isinstance(text, str) or len(text) < 20:
            return []
            
        keywords = []
        
        patterns = {
            'character': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'place': r'\bin [A-Z][a-z]+\b',
            'time': r'\b(19|20)\d{2}\b'
        }
        
        for pattern_type, pattern in patterns.items():
            matches = re.findall(pattern, str(text))
            keywords.extend(matches)
        
        return keywords[:5]
    
    def generate_natural_queries(self, row) -> List[str]:
        """Generate realistic user queries for a movie"""
        queries = []
        
        # Type 1: Genre + Mood/Theme
        if pd.notna(row['genres']) and pd.notna(row['keywords']):
            genres = str(row['genres']).split(', ')
            keywords = str(row['keywords']).split(', ')
            
            if genres and keywords:
                genre = random.choice(genres)
                keyword = random.choice(keywords[:3])
                queries.append(f"{keyword} {genre.lower()} movie")
                queries.append(f"{genre.lower()} film about {keyword}")
        
        # Type 2: Plot-based (semantic)
        if pd.notna(row['plot']) and len(str(row['plot'])) > 50:
            plot = str(row['plot'])
            plot_lower = plot.lower()
            
            if 'love' in plot_lower or 'romance' in plot_lower:
                first_sentence = plot.split('.')[0] if '.' in plot else plot[:100]
                queries.append(f"romantic movie where {first_sentence.lower()}")
            
            if 'space' in plot_lower or 'future' in plot_lower:
                first_sentence = plot.split('.')[0] if '.' in plot else plot[:100]
                queries.append(f"sci-fi about {first_sentence.lower()}")
            
            if 'war' in plot_lower or 'battle' in plot_lower:
                first_sentence = plot.split('.')[0] if '.' in plot else plot[:100]
                queries.append(f"war movie with {first_sentence.lower()}")
        
        # Type 3: Specific details
        if pd.notna(row['plot']):
            entities = self.extract_entities(row['plot'])
            if entities:
                queries.append(f"movie about {random.choice(entities)}")
        
        # Type 4: Vague/semantic
        mood_map = {
            'Horror': ['scary', 'terrifying', 'creepy'],
            'Comedy': ['funny', 'hilarious', 'laugh'],
            'Drama': ['emotional', 'touching', 'deep'],
            'Action': ['intense', 'explosive', 'adrenaline'],
            'Romance': ['romantic', 'love story', 'heartwarming'],
            'Thriller': ['suspenseful', 'tense', 'edge of seat']
        }
        
        if pd.notna(row['genres']):
            for genre, moods in mood_map.items():
                if genre in str(row['genres']):
                    mood = random.choice(moods)
                    queries.append(f"{mood} movie")
                    break
        
        return queries[:3]
    
    def create_movie_text(self, row):
        """Enhanced movie representation"""
        parts = []
        
        if pd.notna(row['title']):
            parts.append(f"{row['title']} {row['title']}")
        
        if pd.notna(row['genres']):
            genres = row['genres']
            parts.append(f"{genres} {genres} {genres}")
        
        if pd.notna(row['keywords']):
            keywords = row['keywords']
            parts.append(f"{keywords} {keywords} {keywords} {keywords}")
        
        if pd.notna(row['plot']):
            plot = str(row['plot'])[:400]
            parts.append(plot)
        
        if pd.notna(row.get('year')) and pd.notna(row.get('rating')):
            parts.append(f"{row['year']} film rated {row['rating']}")
        
        return " ".join(parts)
    
    def generate_positive_pairs(self) -> List[Dict]:
        """Generate all positive training pairs"""
        pairs = []
        
        print("Generating diverse query types...")
        total = len(self.movies_df)
        
        for idx, row in self.movies_df.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx}/{total} movies...")
            
            queries = self.generate_natural_queries(row)
            movie_text = self.create_movie_text(row)
            
            for query in queries:
                if query and len(query) > 5:
                    pairs.append({
                        'query': query,
                        'movie_text': movie_text,
                        'movie_id': row['id'],
                        'movie_title': row['title']
                    })
        
        print(f"Generated {len(pairs)} query-movie pairs")
        return pairs
    
    def save_training_data(self, output_path, max_pairs=15000):
        pairs = self.generate_positive_pairs()
        
        if len(pairs) > max_pairs:
            print(f"Sampling {max_pairs} from {len(pairs)} pairs to prevent overfitting")
            pairs = random.sample(pairs, max_pairs)
        
        df = pd.DataFrame(pairs)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        split_idx = int(len(df) * 0.8)
        train_df = df[:split_idx]
        val_df = df[split_idx:]
        
        train_path = output_path.replace('.csv', '_train.csv')
        val_path = output_path.replace('.csv', '_val.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        print(f"\n{'='*70}")
        print(f" Saved {len(train_df)} training pairs to: {train_path}")
        print(f" Saved {len(val_df)} validation pairs to: {val_path}")
        print(f"{'='*70}\n")
        
        print(f"Sample queries:")
        for i in range(min(5, len(train_df))):
            print(f"  {i+1}. \"{train_df.iloc[i]['query']}\"")
            print(f"     → {train_df.iloc[i]['movie_title']}\n")
        
        return train_df, val_df


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Advanced Training Data Generator")
    print(f"{'='*70}\n")
    
    # Check if movies.csv exists
    movies_path = os.path.join(parent_dir, config.MOVIES_CSV)
    print(f"Looking for movies data at: {movies_path}")
    
    if not os.path.exists(movies_path):
        print(f"\n ERROR: Movies CSV not found at {movies_path}")
        print(f"\nPlease make sure you have:")
        print(f"1. Run data_fetcher.py to download movie data")
        print(f"2. Or place your movies.csv in the data/ folder")
        sys.exit(1)
    
    print(f" Found movies data\n")
    
    # Load movies
    print("Loading movies...")
    df = pd.read_csv(movies_path)
    print(f"Loaded {len(df)} movies\n")
    
    # Generate training data
    generator = AdvancedTrainingDataGenerator(df)
    
    output_path = os.path.join(parent_dir, config.TRAINING_DATA_PATH)
    generator.save_training_data(output_path)
    
    print(f"\n{'='*70}")
    print(f" DONE! Training data ready for fine-tuning")
    print(f"{'='*70}")