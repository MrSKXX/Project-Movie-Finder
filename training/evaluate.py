"""
Module d'évaluation quantitative des modèles
Compare le modèle de base au modèle fine-tuné sur des requêtes de test
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config


class ModelEvaluator:
    """Évalue et compare les performances des modèles"""
    
    def __init__(self):
        """Initialise l'évaluateur avec le dataset et les requêtes de test"""
        movies_path = config.MOVIES_CSV
        
        if not os.path.exists(movies_path):
            print(f"Erreur: {movies_path} introuvable")
            sys.exit(1)
        
        self.movies_df = pd.read_csv(movies_path)
        
        self.test_queries = [
            ("romantic movie on a sinking cruise ship", "Titanic"),
            ("toys that come to life and question existence", "Toy Story"),
            ("sad space movie about isolation", "Solaris"),
            ("AI falls in love with lonely writer", "Her"),
            ("time loop comedy repeat same day", "Groundhog Day"),
            ("underground fight club soap maker", "Fight Club"),
            ("wizard school for children", "Harry Potter"),
            ("dreams within dreams heist", "Inception"),
            ("gladiator seeking revenge in rome", "Gladiator"),
            ("robot cleans earth falls in love", "WALL-E"),
            ("clownfish father searches for son", "Finding Nemo"),
            ("animated movie about emotions inside girl's head", "Inside Out"),
            ("mafia family succession drama", "The Godfather"),
            ("superhero loses powers becomes human", "Spider-Man"),
            ("video game characters escape arcade", "Wreck-It Ralph"),
            ("monster scares children for energy", "Monsters"),
            ("chef rat controls human cooking", "Ratatouille"),
            ("princess with ice powers", "Frozen"),
            ("talking car race", "Cars"),
            ("family of superheroes", "The Incredibles")
        ]
    
    def create_movie_text(self, row):
        """Crée la représentation textuelle d'un film (identique à l'entraînement)"""
        parts = []
        if pd.notna(row['title']): 
            parts.append(f"{row['title']} {row['title']}")
        if pd.notna(row['genres']): 
            parts.append(f"{row['genres']} {row['genres']} {row['genres']}")
        if pd.notna(row['keywords']): 
            parts.append(f"{row['keywords']} {row['keywords']} {row['keywords']} {row['keywords']}")
        if pd.notna(row['plot']): 
            parts.append(str(row['plot'])[:400])
        if pd.notna(row.get('year')) and pd.notna(row.get('rating')): 
            parts.append(f"{row['year']} film rated {row['rating']}")
        return " ".join(parts)
    
    def evaluate_model(self, model_path, model_name):
        """
        Évalue un modèle sur l'ensemble de test
        
        Args:
            model_path: Chemin vers le modèle
            model_name: Nom du modèle pour l'affichage
        
        Returns:
            Dictionnaire avec métriques et résultats détaillés
        """
        print("\n" + "="*70)
        print(f"Évaluation: {model_name}")
        print("="*70 + "\n")
        
        if not model_name.startswith('Base'):
            if not os.path.isabs(model_path):
                model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_path)
        
        try:
            print(f"Chargement du modèle: {model_path}")
            model = SentenceTransformer(model_path)
            print("Modèle chargé avec succès")
        except Exception as e:
            print(f"Échec du chargement: {e}")
            return None
        
        print("\nEncodage des films...")
        movie_texts = self.movies_df.apply(self.create_movie_text, axis=1).tolist()
        embeddings = model.encode(movie_texts, batch_size=64, show_progress_bar=True)
        
        print("Construction de l'index FAISS...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        metrics = {
            'mrr': 0,
            'precision_at_1': 0,
            'precision_at_3': 0,
            'precision_at_5': 0,
            'recall_at_10': 0,
            'avg_rank': []
        }
        
        results_detail = []
        
        print("\nÉvaluation des requêtes...")
        for query, expected_title in self.test_queries:
            q_emb = model.encode([query])
            D, I = index.search(q_emb.astype('float32'), k=10)
            
            top_movies = []
            for idx in I[0]:
                top_movies.append(self.movies_df.iloc[idx]['title'])
            
            rank = None
            for i, title in enumerate(top_movies, 1):
                if expected_title.lower() in title.lower():
                    rank = i
                    break
            
            if rank:
                metrics['mrr'] += 1 / rank
                metrics['avg_rank'].append(rank)
                
                if rank == 1:
                    metrics['precision_at_1'] += 1
                if rank <= 3:
                    metrics['precision_at_3'] += 1
                if rank <= 5:
                    metrics['precision_at_5'] += 1
                if rank <= 10:
                    metrics['recall_at_10'] += 1
            
            results_detail.append({
                'query': query,
                'expected': expected_title,
                'rank': rank if rank else '>10',
                'top_1': top_movies[0],
                'found': 'OK' if rank else 'ÉCHEC'
            })
        
        n = len(self.test_queries)
        metrics['mrr'] /= n
        metrics['precision_at_1'] /= n
        metrics['precision_at_3'] /= n
        metrics['precision_at_5'] /= n
        metrics['recall_at_10'] /= n
        metrics['avg_rank'] = np.mean(metrics['avg_rank']) if metrics['avg_rank'] else None
        
        print(f"\nMÉTRIQUES:")
        print(f"   MRR (Mean Reciprocal Rank): {metrics['mrr']:.3f}")
        print(f"   Precision@1: {metrics['precision_at_1']:.1%}")
        print(f"   Precision@3: {metrics['precision_at_3']:.1%}")
        print(f"   Precision@5: {metrics['precision_at_5']:.1%}")
        print(f"   Recall@10: {metrics['recall_at_10']:.1%}")
        if metrics['avg_rank']:
            print(f"   Rang moyen: {metrics['avg_rank']:.2f}")
        
        print(f"\nRÉSULTATS DÉTAILLÉS:")
        for r in results_detail:
            status = r['found']
            print(f"   [{status}] \"{r['query']}\"")
            print(f"      Attendu: {r['expected']} | Rang: {r['rank']}")
            print(f"      Obtenu: {r['top_1']}\n")
        
        return {
            'model_name': model_name,
            'metrics': metrics,
            'details': results_detail
        }
    
    def compare_models(self):
        """Compare le modèle de base au modèle fine-tuné"""
        models_to_test = {
            'Base (Non entraîné)': 'all-MiniLM-L6-v2',
            'Fine-tuné (v1)': os.path.join(config.MODELS_DIR, 'fine_tuned', 'movie_finder_v1')
        }
        
        all_results = []
        
        for name, path in models_to_test.items():
            result = self.evaluate_model(path, name)
            if result:
                all_results.append(result)
        
        if len(all_results) < 2:
            print("\nImpossible de comparer (un modèle n'a pas pu être chargé)")
            return
        
        print("\n" + "="*70)
        print("COMPARAISON FINALE")
        print("="*70 + "\n")
        
        comparison_data = []
        for r in all_results:
            comparison_data.append({
                'Modèle': r['model_name'],
                'MRR': f"{r['metrics']['mrr']:.3f}",
                'P@1': f"{r['metrics']['precision_at_1']:.1%}",
                'P@3': f"{r['metrics']['precision_at_3']:.1%}",
                'P@5': f"{r['metrics']['precision_at_5']:.1%}",
                'R@10': f"{r['metrics']['recall_at_10']:.1%}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        if len(all_results) == 2:
            base = all_results[0]['metrics']
            trained = all_results[1]['metrics']
            
            mrr_improvement = ((trained['mrr'] - base['mrr']) / base['mrr']) * 100
            p1_improvement = ((trained['precision_at_1'] - base['precision_at_1']) / base['precision_at_1']) * 100 if base['precision_at_1'] > 0 else 0
            
            print(f"\nAMÉLIORATION:")
            print(f"   MRR: +{mrr_improvement:.1f}%")
            print(f"   Precision@1: +{p1_improvement:.1f}%")
        
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, 'evaluation_results.csv')
        comparison_df.to_csv(output_file, index=False)
        print(f"\nRésultats sauvegardés: {output_file}")
        
        return all_results


def main():
    """Point d'entrée pour l'évaluation"""
    print("\n" + "#"*70)
    print("# ÉVALUATION DES MODÈLES")
    print("#"*70 + "\n")
    
    evaluator = ModelEvaluator()
    evaluator.compare_models()
    
    print("\n" + "#"*70)
    print("# ÉVALUATION TERMINÉE")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()