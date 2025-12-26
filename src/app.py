"""
API Flask pour le moteur de recherche sémantique CineSphere
Expose un endpoint REST pour la recherche de films
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from movie_retriever import MovieRetriever
import config

app = Flask(__name__)
CORS(app)

print("\n" + "="*70)
print("INITIALISATION DE CINESPHERE API")
print("="*70 + "\n")

retriever = MovieRetriever(use_trained=True)
retriever.load_movies(config.MOVIES_CSV)

if os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(config.EMBEDDINGS_FILE):
    print("\nChargement de l'index du modèle fine-tuné...")
    retriever.load_index(config.FAISS_INDEX_FILE, config.EMBEDDINGS_FILE)
    print("Index chargé avec succès")
    model_status = "fine-tuned"
else:
    print("\nIndex fine-tuné introuvable")
    print("Veuillez exécuter 'python -m src.movie_retriever' d'abord")
    print("Utilisation du modèle de base en fallback...")
    
    retriever = MovieRetriever(use_trained=False)
    retriever.load_movies(config.MOVIES_CSV)
    
    base_index = config.FAISS_INDEX_FILE.replace('_trained', '')
    base_embeddings = config.EMBEDDINGS_FILE.replace('_trained', '')
    
    if os.path.exists(base_index):
        retriever.load_index(base_index, base_embeddings)
        model_status = "base"
    else:
        print("Aucun index disponible. Veuillez générer un index d'abord.")
        sys.exit(1)

print("\n" + "="*70)
print(f"Serveur prêt - Modèle: {model_status.upper()}")
print("="*70 + "\n")


@app.route('/api/search', methods=['POST'])
def search():
    """
    Endpoint de recherche sémantique
    
    Body JSON:
        query (str): Requête en langage naturel
        top_k (int): Nombre de résultats (défaut: 10)
    
    Returns:
        JSON avec liste de films pertinents
    """
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 10)
    
    if not query:
        return jsonify({'error': 'Requête manquante'}), 400
    
    print(f"Recherche: '{query}'")
    
    results = retriever.search(query, top_k=top_k, adaptive=True)
    
    cleaned_results = []
    for result in results:
        cleaned = {}
        for key, value in result.items():
            if hasattr(value, 'item'):
                cleaned[key] = value.item()
            elif isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    cleaned[key] = None
                else:
                    cleaned[key] = float(value)
            elif isinstance(value, (int, str)):
                cleaned[key] = value
            elif value is None or (isinstance(value, float) and value != value):
                cleaned[key] = None
            else:
                cleaned[key] = str(value) if value else None
        cleaned_results.append(cleaned)
    
    print(f"Retourné: {len(cleaned_results)} résultats")
    if cleaned_results:
        print(f"Top résultat: {cleaned_results[0]['title']} (score: {cleaned_results[0]['final_score']:.3f})")
    
    return jsonify({'results': cleaned_results})


@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint de santé pour vérifier l'état du serveur"""
    return jsonify({
        'status': 'ok',
        'movies_loaded': len(retriever.movies_df),
        'model_type': model_status,
        'message': f'Modèle {model_status} actif'
    })


if __name__ == '__main__':
    print("\nDémarrage du serveur Flask sur http://localhost:5001")
    print("Ouvrez frontend/index.html dans votre navigateur\n")
    app.run(debug=True, port=5001, host='0.0.0.0')