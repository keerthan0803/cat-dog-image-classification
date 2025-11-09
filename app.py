
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, jsonify, render_template, request
import subprocess
import sys
from flask_cors import CORS
from werkzeug.utils import secure_filename
import gdown

MODEL_PATH = 'best_model.h5'
IMG_SIZE = (128, 128)

# Google Drive file ID for the model
# Replace with your model's Google Drive file ID
MODEL_GDRIVE_ID = os.environ.get('MODEL_GDRIVE_ID', '')
MODEL_GDRIVE_URL = f'https://drive.google.com/uc?id={MODEL_GDRIVE_ID}'

# Initialize Flask app first
app = Flask(__name__, template_folder='templates')
CORS(app)

# Lazy load the model to avoid timeout during startup
model = None

def download_model():
    """Download model from Google Drive if not present"""
    if not os.path.exists(MODEL_PATH) and MODEL_GDRIVE_ID:
        print(f"Downloading model from Google Drive...")
        try:
            gdown.download(MODEL_GDRIVE_URL, MODEL_PATH, quiet=False)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

def get_model():
    global model
    if model is None:
        # Download model if not present
        if not os.path.exists(MODEL_PATH):
            download_model()
        
        print("Loading model...")
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully!")
    return model

@app.route('/')
def index():
    print("INDEX ROUTE CALLED")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get model (lazy loading)
        current_model = get_model()
        
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        # Preprocess the image and predict
        img = image.load_img(filepath, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        pred = current_model.predict(img_array, verbose=0)[0][0]
        prediction = 'dog' if pred > 0.5 else 'cat'
        confidence = float(pred) if pred > 0.5 else float(1 - pred)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence * 100, 2)
        })
    except Exception as e:
        # Always return a valid JSON response
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/run-script', methods=['POST'])
def run_script():
    # Use the Python executable from the current environment
    python_exe = sys.executable
    result = subprocess.run([python_exe, 'script.py'], capture_output=True, text=True)
    return jsonify({
        'stdout': result.stdout,
        'stderr': result.stderr,
        'returncode': result.returncode
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)