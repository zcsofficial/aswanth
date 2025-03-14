from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
import face_recognition
from werkzeug.utils import secure_filename
import pickle
from threading import Lock

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['DATABASE_FOLDER'] = 'static/image_database/'
app.config['FEATURES_CACHE'] = 'static/features_cache.pkl'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATABASE_FOLDER'], exist_ok=True)

# Thread lock for cache updates
cache_lock = Lock()

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_or_create_cache():
    """Load cached features or create a new cache."""
    if os.path.exists(app.config['FEATURES_CACHE']):
        with open(app.config['FEATURES_CACHE'], 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    """Save features to cache."""
    with cache_lock:
        with open(app.config['FEATURES_CACHE'], 'wb') as f:
            pickle.dump(cache, f)

def extract_face_features(image_path):
    """Extract facial features using face_recognition library."""
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image, model='large')  # 'large' model for better accuracy
        if not encodings:
            return None
        return encodings[0]  # Return the first face's encoding
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

def compare_faces(known_encoding, unknown_encoding, tolerance=0.6):
    """Compare two face encodings using Euclidean distance."""
    if known_encoding is None or unknown_encoding is None:
        return False
    distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
    return distance < tolerance  # Lower distance = better match

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['DATABASE_FOLDER'], filename)
        file.save(file_path)

        # Update feature cache
        features = extract_face_features(file_path)
        if features is not None:
            cache = load_or_create_cache()
            cache[filename] = features
            save_cache(cache)
        
        return jsonify({'message': 'Image uploaded successfully!'}), 200
    
    return render_template('admin.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'selfie' not in request.files:
        return jsonify({'error': 'No selfie provided'}), 400
    
    file = request.files['selfie']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Extract features from the uploaded selfie
    uploaded_features = extract_face_features(file_path)
    if uploaded_features is None:
        return jsonify({'error': 'No face detected in the uploaded image'}), 400
    
    # Load cached database features
    cache = load_or_create_cache()
    
    # Compare with database images
    matched_images = []
    for db_image, db_features in cache.items():
        if compare_faces(uploaded_features, db_features, tolerance=0.6):
            matched_images.append(db_image)
    
    # If cache is incomplete, scan the database folder
    if not cache or len(cache) < len(os.listdir(app.config['DATABASE_FOLDER'])):
        for db_image in os.listdir(app.config['DATABASE_FOLDER']):
            if db_image in cache:
                continue
            db_image_path = os.path.join(app.config['DATABASE_FOLDER'], db_image)
            db_features = extract_face_features(db_image_path)
            if db_features is not None:
                cache[db_image] = db_features
                if compare_faces(uploaded_features, db_features, tolerance=0.6):
                    matched_images.append(db_image)
        save_cache(cache)
    
    return render_template('result.html', matched_images=matched_images)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)