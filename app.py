from flask import Flask, request, jsonify
from flask_cors import CORS
from movie_retriever import MovieRetriever
import config

app = Flask(__name__)
CORS(app)

print("Initializing Movie Retriever...")
retriever = MovieRetriever()
retriever.load_movies(config.MOVIES_CSV)
retriever.load_index(config.FAISS_INDEX_FILE, config.EMBEDDINGS_FILE)
print("Ready!")

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 10)
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    results = retriever.search(query, top_k=top_k, adaptive=True)
    
    cleaned_results = []
    for result in results:
        cleaned = {}
        for key, value in result.items():
            if hasattr(value, 'item'):
                cleaned[key] = value.item()
            elif isinstance(value, float):
                import math
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
    
    return jsonify({'results': cleaned_results})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'movies_loaded': len(retriever.movies_df)})

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')