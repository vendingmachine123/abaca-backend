from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from datetime import datetime
import logging
from tensorflow.keras.models import load_model
import pickle

# Initialize Flask app
app = Flask(__name__)

# Configure CORS - More permissive for debugging
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # Temporarily allow all origins for debugging
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["*"]
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database simulation
scans_db = []

# Load ML components
model = None
lb = None

def load_ml_components():
    global model, lb
    try:
        model = load_model('output/final_model_full.keras')
        with open('output/label_bin.pickle', 'rb') as f:
            lb = pickle.load(f)
        logger.info("ML components loaded successfully")
    except Exception as e:
        logger.error(f"Error loading ML components: {str(e)}")
        raise e

load_ml_components()

@app.after_request
def after_request(response):
    """Add CORS headers to every response"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response

@app.route('/')
def index():
    return "Abaca Fiber API is running"

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_image():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Process image
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        preds = model.predict(img)
        class_idx = np.argmax(preds)
        class_label = lb.classes_[class_idx]
        confidence = float(preds[0][class_idx])

        # Save to "database"
        scan_data = {
            "id": len(scans_db),
            "grade": class_label,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        scans_db.append(scan_data)

        return jsonify({
            "grade": class_label,
            "confidence": confidence,
            "description": f"This is {class_label} abaca fiber",
            "uses": "Common uses for this grade..."
        })

    except Exception as e:
        logger.error(f"Error in /analyze: {str(e)}")
        return jsonify({"error": str(e)}), 500

def _build_cors_preflight_response():
    response = jsonify({"message": "CORS preflight"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
