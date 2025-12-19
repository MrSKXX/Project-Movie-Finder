import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import os
import sys

# Import config du dossier parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class ModelBenchmark:
    def __init__(self):
        print("Chargement des données...")
        self.df = pd.read_csv(config.MOVIES_CSV)
        # On prépare le texte des films une seule fois
        self.movie_texts = self.df.apply(self._create_text, axis=1).tolist()
        
    def _create_text(self, row):
        # La même logique que pour l'entraînement pour être cohérent
        text = str(row['title'])
        if pd.notna(row['genres']): text += " " + str(row['genres'])
        if pd.notna(row['keywords']): text += " " + str(row['keywords'])
        if pd.notna(row['plot']): text += " " + str(row['plot'])[:300]
        return text

    def evaluate_model(self, model_path, model_name, queries):
        print(f"\n⚡ Test du modèle : {model_name}")
        
        try:
            model = SentenceTransformer(model_path)
        except:
            print(f"❌ Impossible de charger {model_name} ({model_path})")
            return None

        # 1. Encodage de tous les films
        print("   Vectorisation des films...", end="", flush=True)
        start = time.time()
        embeddings = model.encode(self.movie_texts, batch_size=64, show_progress_bar=False)
        print(f" Fait en {time.time()-start:.2f}s")

        # 2. Création Index Faiss
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))

        results = []

        # 3. Lancer les requêtes
        for q in queries:
            q_emb = model.encode([q])
            D, I = index.search(q_emb.astype('float32'), k=3) # Top 3
            
            top_movies = []
            for idx in I[0]:
                top_movies.append(self.df.iloc[idx]['title'])
            
            results.append({
                'Query': q,
                'Top_1': top_movies[0],
                'Top_2': top_movies[1],
                'Top_3': top_movies[2]
            })
            
        return pd.DataFrame(results)

def main():
    benchmark = ModelBenchmark()

    # --- LES REQUÊTES TESTS (Le Juge de Paix) ---
    test_queries = [
        "romantic movie on a cruise ship",   # Test simple
        "toys that come to life",            # Le test qui échouait avant (Toy Story)
        "sad space movie",                   # Test sémantique (Sentiment + Lieu)
        "artificial intelligence love",      # Test thématique
        "time loop repeat day",              # Test concept complexe
        "fighting club soap maker"           # Test ultra précis (Fight Club)
    ]

    # --- LES MODÈLES À COMPARER ---
    models_to_test = {
        "Base (All-MiniLM)": "all-MiniLM-L6-v2",
        "MNRL (1 Epoch)": f"{config.MODELS_DIR}/movie_finder_mnrl_ep1",
        "MNRL (3 Epochs)": f"{config.MODELS_DIR}/movie_finder_mnrl_ep3"
    }

    all_results = pd.DataFrame()

    for name, path in models_to_test.items():
        if name != "Base (All-MiniLM)" and not os.path.exists(path):
            print(f"⚠️ Attention: Le modèle {path} n'existe pas. Avez-vous lancé l'entraînement ?")
            continue
            
        df_res = benchmark.evaluate_model(path, name, test_queries)
        if df_res is not None:
            # On renomme les colonnes pour les fusionner plus tard
            df_res = df_res[['Query', 'Top_1']].rename(columns={'Top_1': f'{name}'})
            
            if all_results.empty:
                all_results = df_res
            else:
                all_results = pd.merge(all_results, df_res, on='Query')

    # Affichage du tableau final
    print("\n" + "="*80)
    print("🏆 RÉSULTATS COMPARATIFS FINAUX")
    print("="*80)
    pd.set_option('display.max_colwidth', 30)
    pd.set_option('display.width', 1000)
    print(all_results)
    
    # Sauvegarde CSV pour le rapport
    output_file = "results/final_benchmark.csv"
    os.makedirs("results", exist_ok=True)
    all_results.to_csv(output_file, index=False)
    print(f"\n📄 Résultats sauvegardés dans {output_file}")

if __name__ == "__main__":
    main()