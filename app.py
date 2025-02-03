from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['DATABASE_FOLDER'] = 'static/image_database/'

# Ensure the upload and database folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATABASE_FOLDER'], exist_ok=True)

# Load pre-trained face detection and recognition models
face_detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",  # Path to prototxt file
    "res10_300x300_ssd_iter_140000.caffemodel"  # Path to pre-trained model
)

face_recognizer = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")  # Pre-trained OpenFace model

def extract_face_features(image_path):
    """Extract facial features from an image using OpenCV's DNN module."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Detect faces
    face_detector.setInput(blob)
    detections = face_detector.forward()

    if len(detections) == 0:
        return None

    # Assume the first detected face is the target
    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    face = image[startY:endY, startX:endX]

    # Extract facial features
    face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    face_recognizer.setInput(face_blob)
    features = face_recognizer.forward()

    return features.flatten()

def compare_faces(features1, features2, threshold=0.6):
    """Compare two facial feature vectors using cosine similarity."""
    if features1 is None or features2 is None:
        return False
    similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return similarity > threshold

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join(app.config['DATABASE_FOLDER'], file.filename)
            file.save(file_path)
            return "Image uploaded successfully!"
    return render_template('admin.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'selfie' not in request.files:
        return redirect(request.url)
    
    file = request.files['selfie']
    if file.filename == '':
        return redirect(request.url)
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Extract features from the uploaded selfie
    uploaded_features = extract_face_features(file_path)
    if uploaded_features is None:
        return "No face detected in the uploaded image."
    
    # Compare with images in the database
    matched_images = []
    for db_image in os.listdir(app.config['DATABASE_FOLDER']):
        db_image_path = os.path.join(app.config['DATABASE_FOLDER'], db_image)
        db_features = extract_face_features(db_image_path)
        
        if db_features is not None and compare_faces(uploaded_features, db_features):
            matched_images.append(db_image)
    
    return render_template('result.html', matched_images=matched_images)

if __name__ == '__main__':
    app.run(debug=True)