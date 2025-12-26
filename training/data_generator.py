"""
Générateur de données d'entraînement avec curriculum learning
Crée des paires (requête, film) à 4 niveaux de difficulté croissante
"""

import pandas as pd
import numpy as np
import random
import re
from typing import List, Dict
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config


class CurriculumDataGenerator:
    """
    Génère des données d'entraînement par curriculum learning
    
    Niveaux:
        1. Compréhension basique des genres
        2. Compréhension thématique
        3. Requêtes basées sur le plot
        4. Requêtes multi-concepts complexes
    """
    
    def __init__(self, movies_df):
        self.movies_df = movies_df
        
        self.genre_descriptors = {
            'Action': [
                'explosive', 'high-octane', 'adrenaline-fueled', 'intense',
                'fight scenes', 'chase sequences', 'battles', 'combat'
            ],
            'Drama': [
                'emotional', 'powerful', 'moving', 'touching', 'deep',
                'character-driven', 'realistic', 'serious', 'profound'
            ],
            'Comedy': [
                'funny', 'hilarious', 'humorous', 'laugh-out-loud', 'witty',
                'lighthearted', 'comedic', 'amusing', 'entertaining'
            ],
            'Horror': [
                'scary', 'terrifying', 'creepy', 'frightening', 'suspenseful',
                'disturbing', 'chilling', 'haunting', 'nightmarish'
            ],
            'Romance': [
                'romantic', 'love story', 'heartwarming', 'passionate',
                'relationship', 'falling in love', 'tender', 'intimate'
            ],
            'Science Fiction': [
                'futuristic', 'sci-fi', 'space', 'technology', 'aliens',
                'dystopian', 'advanced civilization', 'time travel', 'robots'
            ],
            'Thriller': [
                'suspenseful', 'tense', 'gripping', 'edge-of-seat', 'mystery',
                'psychological', 'intense', 'nail-biting', 'twisting'
            ],
            'Fantasy': [
                'magical', 'mythical', 'enchanted', 'supernatural', 'epic',
                'wizards', 'dragons', 'mythological', 'otherworldly'
            ],
            'Animation': [
                'animated', 'cartoon', 'family-friendly', 'colorful',
                'for kids', 'CGI', 'hand-drawn', 'Pixar-style'
            ],
            'Adventure': [
                'thrilling journey', 'quest', 'exploration', 'expedition',
                'treasure hunt', 'discovery', 'daring', 'heroic'
            ]
        }
        
        self.themes = {
            'revenge': ['seeking revenge', 'vendetta', 'payback', 'retribution'],
            'friendship': ['bond between friends', 'friendship', 'companions', 'allies'],
            'family': ['family bonds', 'parent-child', 'siblings', 'family drama'],
            'survival': ['fight for survival', 'staying alive', 'survival against odds'],
            'redemption': ['seeking redemption', 'second chance', 'atonement'],
            'coming-of-age': ['growing up', 'loss of innocence', 'self-discovery'],
            'war': ['wartime', 'battlefield', 'military conflict', 'soldiers'],
            'crime': ['criminal underworld', 'heist', 'organized crime', 'gangsters'],
            'identity': ['finding oneself', 'identity crisis', 'who am I'],
            'sacrifice': ['ultimate sacrifice', 'giving up everything', 'selflessness']
        }
        
        self.character_patterns = {
            'hero': ['protagonist', 'hero', 'main character', 'champion'],
            'antihero': ['flawed protagonist', 'morally grey hero', 'reluctant hero'],
            'villain': ['antagonist', 'villain', 'evil mastermind', 'bad guy'],
            'mentor': ['wise guide', 'teacher', 'mentor figure', 'guru'],
            'sidekick': ['loyal companion', 'best friend', 'partner', 'ally'],
            'underdog': ['unlikely hero', 'underdog', 'against all odds']
        }
        
        self.plot_patterns = [
            'rags to riches story',
            'hero\'s journey',
            'tragedy',
            'rebirth and transformation',
            'quest narrative',
            'voyage and return',
            'overcoming the monster',
            'fish out of water'
        ]
        
        self.settings = {
            'urban': ['city', 'metropolitan', 'urban setting', 'downtown'],
            'rural': ['countryside', 'small town', 'rural', 'village'],
            'space': ['outer space', 'spaceship', 'alien planet', 'galaxy'],
            'historical': ['period piece', 'historical setting', 'based on true events'],
            'fantasy_world': ['fantasy realm', 'magical world', 'mythical land'],
            'underwater': ['under the sea', 'ocean depths', 'underwater world'],
            'post_apocalyptic': ['after apocalypse', 'dystopian future', 'wasteland']
        }
    
    def create_movie_text(self, row):
        """
        Crée la représentation textuelle d'un film
        Doit être identique à celle utilisée dans movie_retriever.py
        """
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
    
    def extract_plot_elements(self, plot_text: str) -> Dict:
        """Extrait les éléments sémantiques du résumé"""
        if not isinstance(plot_text, str):
            return {}
        
        plot_lower = plot_text.lower()
        elements = {
            'themes': [],
            'settings': [],
            'character_types': []
        }
        
        for theme, indicators in self.themes.items():
            if any(ind in plot_lower for ind in indicators):
                elements['themes'].append(theme)
        
        for setting, indicators in self.settings.items():
            if any(ind in plot_lower for ind in indicators):
                elements['settings'].append(setting)
        
        return elements
    
    def generate_level1_genre_queries(self, row) -> List[str]:
        """Niveau 1: Compréhension basique des genres"""
        queries = []
        
        if pd.notna(row['genres']):
            genres = str(row['genres']).split(', ')
            for genre in genres:
                if genre in self.genre_descriptors:
                    descriptors = self.genre_descriptors[genre]
                    for desc in random.sample(descriptors, min(2, len(descriptors))):
                        queries.append(f"{desc} {genre.lower()} movie")
                        queries.append(f"{desc} film")
        
        return queries
    
    def generate_level2_theme_queries(self, row) -> List[str]:
        """Niveau 2: Compréhension thématique"""
        queries = []
        
        if pd.notna(row['plot']):
            elements = self.extract_plot_elements(row['plot'])
            
            for theme in elements['themes'][:2]:
                queries.append(f"movie about {theme}")
                
                if pd.notna(row['genres']):
                    genre = str(row['genres']).split(', ')[0]
                    queries.append(f"{genre.lower()} movie about {theme}")
        
        return queries
    
    def generate_level3_plot_based_queries(self, row) -> List[str]:
        """Niveau 3: Requêtes basées sur le plot"""
        queries = []
        
        if not pd.notna(row['plot']) or len(str(row['plot'])) < 50:
            return queries
        
        plot = str(row['plot'])
        plot_lower = plot.lower()
        
        sentences = [s.strip() for s in plot.split('.') if len(s.strip()) > 20]
        
        if sentences:
            first_sent = sentences[0].lower()
            
            if ' when ' in first_sent:
                parts = first_sent.split(' when ')
                if len(parts) == 2:
                    queries.append(f"movie where {parts[1][:50]}")
            
            if ' after ' in first_sent:
                parts = first_sent.split(' after ')
                if len(parts) == 2:
                    queries.append(f"movie about {parts[0][:50]} after {parts[1][:50]}")
            
            if first_sent.startswith(('a ', 'an ', 'the ')):
                words = first_sent.split()
                if len(words) > 5:
                    subject = ' '.join(words[:5])
                    queries.append(f"movie about {subject}")
        
        plot_indicators = {
            'falls in love': 'love story',
            'seeks revenge': 'revenge story',
            'discovers': 'discovery story',
            'must save': 'rescue mission',
            'fights': 'action story',
            'travels': 'journey story',
            'escape': 'escape story'
        }
        
        for indicator, story_type in plot_indicators.items():
            if indicator in plot_lower:
                context = plot_lower.split(indicator)[0].split()[-5:]
                queries.append(f"{' '.join(context)} {indicator}")
        
        return queries
    
    def generate_level4_multimodal_queries(self, row) -> List[str]:
        """Niveau 4: Requêtes multi-concepts complexes"""
        queries = []
        
        if pd.notna(row['genres']) and pd.notna(row['plot']):
            genre = str(row['genres']).split(', ')[0]
            elements = self.extract_plot_elements(row['plot'])
            
            if elements['settings'] and elements['themes']:
                setting = elements['settings'][0]
                theme = elements['themes'][0]
                queries.append(f"{genre.lower()} movie set in {setting} about {theme}")
        
        if pd.notna(row['genres']) and pd.notna(row['keywords']):
            genre = str(row['genres']).split(', ')[0]
            keywords = str(row['keywords']).split(', ')
            
            if len(keywords) >= 2:
                kw1, kw2 = random.sample(keywords[:5], 2)
                queries.append(f"{genre.lower()} with {kw1} and {kw2}")
        
        return queries
    
    def generate_comprehensive_pairs(self) -> List[Dict]:
        """Génère toutes les paires d'entraînement par curriculum"""
        all_pairs = []
        
        print("Génération Niveau 1: Requêtes de genre...")
        level1_count = 0
        
        print("Génération Niveau 2: Requêtes thématiques...")
        level2_count = 0
        
        print("Génération Niveau 3: Requêtes basées sur le plot...")
        level3_count = 0
        
        print("Génération Niveau 4: Requêtes multi-concepts...")
        level4_count = 0
        
        for idx, row in self.movies_df.iterrows():
            if idx % 200 == 0:
                print(f"Traitement: {idx}/{len(self.movies_df)} films...")
            
            movie_text = self.create_movie_text(row)
            
            l1_queries = self.generate_level1_genre_queries(row)
            for q in l1_queries:
                all_pairs.append({
                    'query': q,
                    'movie_text': movie_text,
                    'level': 1,
                    'movie_id': row['id'],
                    'movie_title': row['title']
                })
                level1_count += 1
            
            l2_queries = self.generate_level2_theme_queries(row)
            for q in l2_queries:
                all_pairs.append({
                    'query': q,
                    'movie_text': movie_text,
                    'level': 2,
                    'movie_id': row['id'],
                    'movie_title': row['title']
                })
                level2_count += 1
            
            l3_queries = self.generate_level3_plot_based_queries(row)
            for q in l3_queries:
                all_pairs.append({
                    'query': q,
                    'movie_text': movie_text,
                    'level': 3,
                    'movie_id': row['id'],
                    'movie_title': row['title']
                })
                level3_count += 1
            
            l4_queries = self.generate_level4_multimodal_queries(row)
            for q in l4_queries:
                all_pairs.append({
                    'query': q,
                    'movie_text': movie_text,
                    'level': 4,
                    'movie_id': row['id'],
                    'movie_title': row['title']
                })
                level4_count += 1
        
        print(f"\nDistribution des requêtes:")
        print(f"   Niveau 1 (Genre): {level1_count}")
        print(f"   Niveau 2 (Thème): {level2_count}")
        print(f"   Niveau 3 (Plot): {level3_count}")
        print(f"   Niveau 4 (Multi-concept): {level4_count}")
        print(f"   Total: {len(all_pairs)}")
        
        return all_pairs
    
    def save_training_data(self, output_path, max_pairs=20000):
        """
        Génère et sauvegarde les données d'entraînement
        
        Args:
            output_path: Chemin de base pour les fichiers train/val
            max_pairs: Nombre maximum de paires (échantillonnage si dépassé)
        """
        pairs = self.generate_comprehensive_pairs()
        
        if len(pairs) > max_pairs:
            df = pd.DataFrame(pairs)
            sampled = df.groupby('level', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max_pairs // 4)),
                include_groups=False
            ).reset_index(drop=True)
            pairs = sampled.to_dict('records')
        
        df = pd.DataFrame(pairs)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        split_idx = int(len(df) * 0.8)
        train_df = df[:split_idx]
        val_df = df[split_idx:]
        
        train_path = output_path.replace('.csv', '_train.csv')
        val_path = output_path.replace('.csv', '_val.csv')
        
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        print(f"\n{len(train_df)} paires d'entraînement -> {train_path}")
        print(f"{len(val_df)} paires de validation -> {val_path}")
        
        print(f"\nÉchantillon de requêtes:")
        for level in [1, 2, 3, 4]:
            sample = train_df[train_df['level'] == level].sample(min(2, len(train_df[train_df['level'] == level])))
            print(f"\n  Niveau {level}:")
            for _, row in sample.iterrows():
                print(f"    \"{row['query']}\" -> {row['movie_title']}")
        
        return train_df, val_df


def main():
    """Point d'entrée pour générer les données d'entraînement"""
    print("\n" + "="*70)
    print("GÉNÉRATEUR DE DONNÉES PAR CURRICULUM LEARNING")
    print("="*70 + "\n")
    
    movies_path = config.MOVIES_CSV
    
    if not os.path.exists(movies_path):
        print(f"Erreur: {movies_path} introuvable")
        print("Exécutez d'abord: python -m src.data_fetcher")
        sys.exit(1)
    
    df = pd.read_csv(movies_path)
    print(f"Chargé: {len(df)} films\n")
    
    generator = CurriculumDataGenerator(df)
    generator.save_training_data(config.TRAINING_DATA_PATH)
    
    print("\n" + "="*70)
    print("Données d'entraînement prêtes")
    print("="*70)


if __name__ == "__main__":
    main()
    