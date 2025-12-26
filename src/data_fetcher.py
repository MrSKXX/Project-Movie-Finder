"""
Module de récupération des données depuis l'API TMDB
Télécharge les métadonnées des films populaires
"""

import requests
import pandas as pd
import time
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class TMDbFetcher:
    """Récupère les données de films depuis l'API TMDB"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = config.TMDB_BASE_URL
        self.session = requests.Session()
        
    def fetch_popular_movies(self, num_pages=50):
        """
        Récupère les films populaires sur plusieurs pages
        
        Args:
            num_pages: Nombre de pages à récupérer (20 films par page)
        
        Returns:
            Liste de dictionnaires contenant les métadonnées des films
        """
        movies = []
        
        print(f"Récupération de {num_pages} pages de films populaires...")
        for page in tqdm(range(1, num_pages + 1)):
            try:
                url = f"{self.base_url}/movie/popular"
                params = {
                    'api_key': self.api_key,
                    'page': page,
                    'language': 'en-US'
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for movie in data.get('results', []):
                    movie_details = self.fetch_movie_details(movie['id'])
                    if movie_details:
                        movies.append(movie_details)
                
                time.sleep(0.25)
                
            except Exception as e:
                print(f"\nErreur page {page}: {e}")
                continue
                
        return movies
    
    def fetch_movie_details(self, movie_id):
        """
        Récupère les détails complets d'un film
        
        Args:
            movie_id: Identifiant TMDB du film
        
        Returns:
            Dictionnaire avec titre, plot, genres, keywords, etc.
        """
        try:
            url = f"{self.base_url}/movie/{movie_id}"
            params = {
                'api_key': self.api_key,
                'append_to_response': 'keywords',
                'language': 'en-US'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                'id': data.get('id'),
                'title': data.get('title'),
                'plot': data.get('overview', ''),
                'genres': ', '.join([g['name'] for g in data.get('genres', [])]),
                'keywords': ', '.join([k['name'] for k in data.get('keywords', {}).get('keywords', [])]),
                'year': data.get('release_date', '')[:4] if data.get('release_date') else '',
                'rating': data.get('vote_average', 0),
                'popularity': data.get('popularity', 0),
                'poster_path': data.get('poster_path', '')
            }
            
        except Exception:
            return None
    
    def save_to_csv(self, movies, filename):
        """
        Sauvegarde les films dans un fichier CSV
        
        Args:
            movies: Liste de dictionnaires de films
            filename: Chemin du fichier de sortie
        
        Returns:
            DataFrame pandas des films sauvegardés
        """
        df = pd.DataFrame(movies)
        df = df[df['plot'].str.len() > 20]
        df = df.drop_duplicates(subset=['id'])
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        
        print(f"\n{len(df)} films sauvegardés dans {filename}")
        return df


def main():
    """Point d'entrée pour récupérer les données TMDB"""
    fetcher = TMDbFetcher(config.TMDB_API_KEY)
    movies = fetcher.fetch_popular_movies(num_pages=200)
    df = fetcher.save_to_csv(movies, config.MOVIES_CSV)
    
    print(f"\nRésumé du dataset:")
    print(f"Total films: {len(df)}")
    print(f"\nÉchantillon:")
    print(df[['title', 'year', 'genres']].head())


if __name__ == "__main__":
    main()