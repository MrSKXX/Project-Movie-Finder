"""
Interface en ligne de commande pour CineSphere
Point d'entrée principal pour la recherche interactive de films
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.movie_retriever import MovieRetriever
from src import config


class CineSphere:
    """Interface CLI pour la recherche sémantique de films"""
    
    def __init__(self):
        print("Initialisation de CineSphere...")
        self.retriever = MovieRetriever(use_trained=True)
        self.retriever.load_movies(config.MOVIES_CSV)
        
        if os.path.exists(config.FAISS_INDEX_FILE):
            self.retriever.load_index(config.FAISS_INDEX_FILE, config.EMBEDDINGS_FILE)
        else:
            print("\nIndex FAISS introuvable. Génération en cours...")
            embeddings = self.retriever.generate_embeddings()
            self.retriever.embeddings = embeddings
            self.retriever.build_index(embeddings)
            self.retriever.save_index(config.FAISS_INDEX_FILE, config.EMBEDDINGS_FILE)
        
        print("Prêt!\n")
    
    def display_results(self, results):
        """
        Affiche les résultats de recherche formatés
        
        Args:
            results: Liste de dictionnaires de films
        """
        print("\n" + "="*70)
        print(f"RECOMMANDATIONS ({len(results)} résultats)")
        print("="*70 + "\n")
        
        for i, movie in enumerate(results, 1):
            print(f"{i}. {movie['title']} ({movie['year']})")
            print(f"   Genres: {movie['genres']}")
            print(f"   Note: {movie['rating']:.1f}/10")
            print(f"   Score de pertinence: {movie['final_score']:.1%}")
            
            if movie['plot'] and len(str(movie['plot'])) > 10:
                plot = str(movie['plot'])
                if len(plot) > 150:
                    plot = plot[:150] + "..."
                print(f"   Résumé: {plot}")
            
            print()
    
    def run(self):
        """Boucle principale de l'interface CLI"""
        print("="*70)
        print("CINESPHERE - Recherche Sémantique de Films")
        print("="*70)
        print("\nEntrez votre requête en langage naturel")
        print("Exemple: 'film romantique sur un bateau'")
        print("Tapez 'quit' ou 'exit' pour quitter\n")
        
        while True:
            try:
                query = input("Que recherchez-vous? > ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nMerci d'avoir utilisé CineSphere!")
                    break
                
                if not query:
                    continue
                
                results = self.retriever.search(query, top_k=10, adaptive=True)
                
                if not results:
                    print("\nAucun résultat trouvé. Essayez une autre requête.\n")
                    continue
                
                self.display_results(results)
                
            except KeyboardInterrupt:
                print("\n\nMerci d'avoir utilisé CineSphere!")
                break
            except Exception as e:
                print(f"Erreur: {e}")


def main():
    """Point d'entrée principal"""
    try:
        app = CineSphere()
        app.run()
    except Exception as e:
        print(f"Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()