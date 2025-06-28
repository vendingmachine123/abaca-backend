from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import pickle
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app, resources={
    r"/analyze": {"origins": ["https://abaca.vercel.app"]},
    r"/history": {"origins": ["https://abaca.vercel.app"]},
    r"/scan/*": {"origins": ["https://abaca.vercel.app"]},
    r"/api/grade-info": {"origins": ["https://abaca.vercel.app"]}
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory "database" for demonstration
scans_db = []

# Load the trained model and label binarizer
model = None
lb = None

def load_ml_components():
    global model, lb
    try:
        model = load_model('output/final_model_full.keras')
        with open('output/label_bin.pickle', 'rb') as f:
            lb = pickle.load(f)
        print("Model and label binarizer loaded successfully")
    except Exception as e:
        print(f"Error loading ML components: {str(e)}")
        raise e

# Load ML components when the app starts
load_ml_components()

@app.route('/favicon.ico')
def favicon():
    return '', 404

@app.route('/')
def index():
    return "API is running"

# API endpoint to get scan history
@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(scans_db)

# API endpoint to get specific scan details
@app.route('/scan/<scan_id>', methods=['GET'])
def get_scan(scan_id):
    try:
        scan_id_int = int(scan_id)
        if 0 <= scan_id_int < len(scans_db):
            return jsonify(scans_db[scan_id_int])
        else:
            return jsonify({'error': 'Scan not found'}), 404
    except ValueError:
        return jsonify({'error': 'Invalid scan ID'}), 400

# API endpoint to get grade information
@app.route('/api/grade-info', methods=['GET'])
def get_grade_info():
    grade_info = {
        "grades": [
            {
                "grade": "Grade 1",
                "description": "Grade 1 abaca fibers are the highest quality, longest, and most durable fibers.",
                "characteristics": [
                    "Longest fibers (7-9 feet)",
                    "Cleanest and most uniform",
                    "Highest tensile strength"
                ]
            },
            {
                "grade": "Grade 2",
                "description": "Grade 2 abaca fibers are high quality but slightly shorter than Grade 1.",
                "characteristics": [
                    "Medium-long fibers (5-7 feet)",
                    "Slightly less uniform than Grade 1",
                    "High tensile strength"
                ]
            },
            {
                "grade": "Grade 3",
                "description": "Grade 3 abaca fibers are medium quality, suitable for many applications.",
                "characteristics": [
                    "Medium fibers (3-5 feet)",
                    "Some variability in quality",
                    "Good tensile strength"
                ]
            },
            {
                "grade": "Grade 4",
                "description": "Grade 4 abaca fibers are shorter and less durable, used for lower-end products.",
                "characteristics": [
                    "Shortest fibers (1-3 feet)",
                    "Most variable in quality",
                    "Lower tensile strength"
                ]
            }
        ],
        "common_uses": {
            "Grade 1": [
                "High-quality ropes",
                "Specialty papers",
                "Currency paper",
                "Tea bags"
            ],
            "Grade 2": [
                "Medium-quality ropes",
                "Handicrafts",
                "Filters",
                "Decorative papers"
            ],
            "Grade 3": [
                "Lower-grade ropes",
                "Mixed fiber products",
                "Packaging materials",
                "Industrial applications"
            ],
            "Grade 4": [
                "Low-grade products",
                "Mixed fiber applications",
                "Stuffing materials",
                "Composting"
            ]
        }
    }
    return jsonify(grade_info)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        image_data = file.read()
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': 'Could not process image'}), 400
        
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = lb.classes_[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Save the scan to our "database"
        scan_data = {
            'id': len(scans_db),
            'grade': predicted_class,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'image_path': f"scan_{len(scans_db)}.jpg"
        }
        scans_db.append(scan_data)
        
        grade_info = {
            'Grade 1': {
                'description': 'Grade 1 abaca fibers are the highest quality, longest, and most durable fibers.',
                'uses': 'High-quality ropes, specialty papers, currency'
            },
            'Grade 2': {
                'description': 'Grade 2 abaca fibers are high quality but slightly shorter than Grade 1.',
                'uses': 'Medium-quality ropes, tea bags, filters'
            },
            'Grade 3': {
                'description': 'Grade 3 abaca fibers are medium quality, suitable for many applications.',
                'uses': 'Handicrafts, lower-grade ropes'
            },
            'Grade 4': {
                'description': 'Grade 4 abaca fibers are shorter and less durable, used for lower-end products.',
                'uses': 'Low-grade products, mixed fiber applications'
            }
        }
        
        return jsonify({
            'grade': predicted_class,
            'confidence': confidence,
            'description': grade_info.get(predicted_class, {}).get('description', ''),
            'uses': grade_info.get(predicted_class, {}).get('uses', '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

def preprocess_image(image_data):
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None
        
        image = cv2.resize(image, (128, 128))
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
