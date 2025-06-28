from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import pickle
from flask_cors import CORS
from datetime import datetime
import logging

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["https://abaca.vercel.app", "http://localhost:*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory "database" for demonstration
scans_db = []

# Load the trained model and label binarizer
model = None
lb = None

def load_ml_components():
    """Load the machine learning model and label binarizer"""
    global model, lb
    try:
        model = load_model('output/final_model_full.keras')
        with open('output/label_bin.pickle', 'rb') as f:
            lb = pickle.load(f)
        logger.info("Model and label binarizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading ML components: {str(e)}")
        raise e

# Load ML components when the app starts
load_ml_components()

@app.after_request
def after_request(response):
    """Add CORS headers to every response"""
    response.headers.add('Access-Control-Allow-Origin', 'https://abaca.vercel.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.route('/favicon.ico')
def favicon():
    return '', 404

@app.route('/')
def index():
    return "Abaca Fiber Grade Classifier API is running"

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

# API endpoint to get scan history
@app.route('/history', methods=['GET', 'OPTIONS'])
def get_history():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    return jsonify(scans_db)

# API endpoint to get specific scan details
@app.route('/scan/<scan_id>', methods=['GET', 'OPTIONS'])
def get_scan(scan_id):
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    try:
        scan_id_int = int(scan_id)
        if 0 <= scan_id_int < len(scans_db):
            return jsonify(scans_db[scan_id_int])
        return jsonify({'error': 'Scan not found'}), 404
    except ValueError:
        return jsonify({'error': 'Invalid scan ID'}), 400

# API endpoint to get grade information
@app.route('/api/grade-info', methods=['GET', 'OPTIONS'])
def get_grade_info():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
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

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_image():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Validate file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
            
        image_data = file.read()
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': 'Could not process image'}), 400
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = lb.classes_[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Save the scan to database
        scan_data = {
            'id': len(scans_db),
            'grade': predicted_class,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'image_path': f"scan_{len(scans_db)}.jpg"
        }
        scans_db.append(scan_data)
        
        # Prepare response with grade information
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
        
        response_data = {
            'grade': predicted_class,
            'confidence': confidence,
            'description': grade_info.get(predicted_class, {}).get('description', ''),
            'uses': grade_info.get(predicted_class, {}).get('uses', '')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_data):
    """Preprocess the image for model prediction"""
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            logger.error("Could not decode image")
            return None
        
        # Resize and normalize image
        image = cv2.resize(image, (128, 128))
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def _build_cors_preflight_response():
    """Build a response for CORS preflight requests"""
    response = jsonify({'message': 'CORS preflight'})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_response(response):
    """Add CORS headers to a response"""
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
