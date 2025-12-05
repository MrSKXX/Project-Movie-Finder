import sys
from movie_retriever import MovieRetriever
import config

class MovieFinder:
    def __init__(self):
        print("Initializing AI Movie Finder...")
        self.retriever = MovieRetriever()
        self.retriever.load_movies(config.MOVIES_CSV)
        self.retriever.load_index(config.FAISS_INDEX_FILE, config.EMBEDDINGS_FILE)
        print("Ready!\n")
    
    def display_results(self, results):
        print("\n" + "="*70)
        print(f"TOP RECOMMENDATIONS ({len(results)} matches):")
        print("="*70 + "\n")
        
        for i, movie in enumerate(results, 1):
            print(f"{i}. {movie['title']} ({movie['year']})")
            print(f"   Genres: {movie['genres']}")
            print(f"   Rating: ⭐ {movie['rating']:.1f}/10")
            print(f"   Match Score: {movie['final_score']:.1%}")
            
            if movie['plot'] and len(str(movie['plot'])) > 10:
                plot = str(movie['plot'])
                if len(plot) > 150:
                    plot = plot[:150] + "..."
                print(f"   Plot: {plot}")
            
            print()
    
    def run(self):
        print("🎬 AI Movie Finder")
        print("Type your query in natural language (e.g., 'romance on a cruise ship')")
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                query = input("What kind of movie are you looking for? > ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Thanks for using AI Movie Finder!")
                    break
                
                if not query:
                    continue
                
                results = self.retriever.search(query, top_k=10, adaptive=True)
                
                if not results:
                    print("\n❌ No good matches found. Try a different query.\n")
                    continue
                
                self.display_results(results)
                
            except KeyboardInterrupt:
                print("\n\nThanks for using AI Movie Finder!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    finder = MovieFinder()
    finder.run()

if __name__ == "__main__":
    main()