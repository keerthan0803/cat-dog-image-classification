import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model once at startup
MODEL_PATH = 'model.h5'
IMG_SIZE = (128, 128)
model = load_model(MODEL_PATH)

from flask import Flask, jsonify, render_template, request
import subprocess
import sys
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)
    # Preprocess the image and predict
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    prediction = 'dog' if pred > 0.5 else 'cat'
    os.remove(filepath)
    return jsonify({'prediction': prediction})

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

if __name__ == '__main__':
    app.run(debug=True)