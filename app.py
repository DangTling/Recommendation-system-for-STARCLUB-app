from flask import Flask, request, jsonify
import numpy as np
from qdrant_client import QdrantClient
# from qdrant_client.http.models import PointIdsList
from sentence_transformers import SentenceTransformer
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt
import datetime
import os
import json
from dotenv import load_dotenv
from functools import wraps

load_dotenv()

app = Flask(__name__)

app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(days=1)
users_json = os.getenv('USERS')
USERS = json.loads(users_json)

jwt = JWTManager(app)

def role_required(roles):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            jwt_data = get_jwt()
            user_role = jwt_data['sub']['role']
            if user_role not in roles:
                return jsonify({"error": "Unauthorized access"}), 403
            return function(*args, **kwargs)
        return wrapper
    return decorator

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    if username not in USERS or USERS[username]['password'] != password:
        return jsonify({"error": "Invalid username or password"}), 401

    user_role = USERS[username]['role']
    access_token = create_access_token(identity={"username": username, "role": user_role})
    return jsonify(access_token=access_token), 200

qdrant = QdrantClient(url='http://localhost:6333')  
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/search', methods=['POST'])
@jwt_required()
@role_required(['user', 'admin'])
def search_songs():
    
    song_title = request.json.get('title')
    song_artist = request.json.get('artist')
    song_category = request.json.get('category')
    song_description = request.json.get('description')
    top_k = request.json.get('top_k', 5) 

    # query_description = f"{song_title} - {song_artist} - {song_category} - {song_description}"
    # embedding = model.encode([query_description])[0]

    title_artist_vector = model.encode(song_title + " " + song_artist)
    category_vector = model.encode(song_category)
    description_vector = model.encode(song_description)

    query_vector = np.hstack([title_artist_vector, category_vector, description_vector])

    results = qdrant.search(collection_name='songs_collection_database', query_vector=query_vector.tolist(), limit=top_k)

    recommendations = [res.payload for res in results]
 
    return jsonify(recommendations)

@app.route('/search-in-home', methods=['POST'])
@jwt_required()
@role_required(['user', 'admin'])
def search_songs_in_home():
    song_list = request.json.get('songs')
    top_k = request.json.get('top_k', 10)

    if not song_list:
        return jsonify({"error": "Song list is required"}), 400

    embeddings = []
    for song in song_list:
        song_title = song.get('title')
        song_artist = song.get('artist')
        song_category = song.get('category')
        song_description = song.get('description')

        # query_description = f"{song_title} - {song_artist} - {song_category} - {song_description}"
        
        # embedding = model.encode([query_description])[0]
        title_artist_vector = model.encode(song_title + " " + song_artist)
        category_vector = model.encode(song_category)
        description_vector = model.encode(song_description)

        query_vector = np.hstack([title_artist_vector, category_vector, description_vector])
        embeddings.append(query_vector)

    if embeddings:
        average_embedding = sum(embeddings) / len(embeddings)
    else:
        return jsonify({"error": "No embeddings were calculated"}), 400

    results = qdrant.search(
        collection_name='songs_collection_database',
        query_vector=average_embedding.tolist(),
        limit=top_k
    )

    recommendations = [res.payload for res in results]

    return jsonify(recommendations)

@app.route('/add_song', methods=['POST'])
@jwt_required()
@role_required(['user', 'admin'])
def add_song():
    song_title = request.json.get('title')
    song_artist = request.json.get('artist')
    song_category = request.json.get('category')
    song_description = request.json.get('description')

    song_id = request.json.get('id')

    if not all([song_title, song_artist, song_category, song_description]):
        return jsonify({"error": "Missing required fields"}), 400

    # description = f"{song_title} - {song_artist} - {song_category} - {song_description}"
    # embedding = model.encode([description])[0]

    title_artist_vector = model.encode(song_title + " " + song_artist)
    category_vector = model.encode(song_category)
    description_vector = model.encode(song_description)

    query_vector = np.hstack([title_artist_vector, category_vector, description_vector])

    qdrant.upsert(
        collection_name='songs_collection_database',
        points=[
            {
                'id': song_id, 
                'vector': query_vector.tolist(),
                'payload': {
                    'title': song_title,
                    'artist_name': song_artist,
                    'category_name': song_category,
                    'description': song_description,
                    'lyrics': "",
                    'id': str(song_id)
                }
            }
        ]
    )

    return jsonify({"message": "Song added successfully"}), 200

@app.route('/delete', methods=['POST'])
@jwt_required()
@role_required(['admin'])
def delete_song():
    vector_id = request.json.get('id')

    if not vector_id:
        return jsonify({"error": "ID is required"}), 400 

    qdrant.delete(
        collection_name='songs_collection_database',
        points_selector=[vector_id]
    )

    return jsonify({"message": f"Vector with ID {vector_id} has been deleted."}), 200

@app.route('/vectors', methods=['GET'])
@jwt_required()
@role_required(['admin'])
def get_all_vectors():
    vectors = qdrant.scroll(
        collection_name='songs_collection_database', 
        scroll_filter=None,  
        limit=100 
    )
    
    return jsonify([vector.payload for vector in vectors[0]]), 200

if __name__ == '__main__':
    app.run(debug=True)
