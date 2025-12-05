import requests
import pandas as pd
import time
from tqdm import tqdm
import config
import os

class TMDbFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = config.TMDB_BASE_URL
        self.session = requests.Session()
        
    def fetch_popular_movies(self, num_pages=50):
        movies = []
        
        print(f"Fetching {num_pages} pages of popular movies...")
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
                print(f"\nError fetching page {page}: {e}")
                continue
                
        return movies
    
    def fetch_movie_details(self, movie_id):
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
            
        except Exception as e:
            return None
    
    def save_to_csv(self, movies, filename):
        df = pd.DataFrame(movies)
        df = df[df['plot'].str.len() > 20]
        df = df.drop_duplicates(subset=['id'])
        df.to_csv(filename, index=False)
        print(f"\nSaved {len(df)} movies to {filename}")
        return df

def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    fetcher = TMDbFetcher(config.TMDB_API_KEY)
    movies = fetcher.fetch_popular_movies(num_pages=200)
    df = fetcher.save_to_csv(movies, config.MOVIES_CSV)
    
    print(f"\nDataset summary:")
    print(f"Total movies: {len(df)}")
    print(f"\nSample:")
    print(df[['title', 'year', 'genres']].head())

if __name__ == "__main__":
    main()