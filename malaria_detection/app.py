"""
Web Dashboard for Malaria Detection System
Displays training results and allows real-time predictions
Run: python app.py
"""

from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import os
import json
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from prediction import MalariaDiagnoser
import atexit
import webbrowser
import threading
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
diagnoser = None
results_path = Path('results')
models_path = Path('models')

def init_app():
    """Initialize the app"""
    global diagnoser
    
    # Try to load the trained model
    model_path = models_path / 'best_model.h5'
    
    if model_path.exists():
        try:
            diagnoser = MalariaDiagnoser(str(model_path))
            print("[OK] Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            diagnoser = None
    else:
        print("Note: No trained model found. Train first with: python main.py")
        diagnoser = None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """Get system status"""
    return jsonify({
        'model_exists': (models_path / 'best_model.h5').exists(),
        'model_loaded': diagnoser is not None,
        'results_exist': results_path.exists(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/results')
def get_results():
    """Get training results"""
    try:
        results = {
            'training_history': None,
            'confusion_matrix': None,
            'report': None,
            'images': []
        }
        
        # Check for result images
        if results_path.exists():
            for img_file in results_path.glob('*.png'):
                with open(img_file, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                    results['images'].append({
                        'name': img_file.stem,
                        'src': f'data:image/png;base64,{img_data}'
                    })
            
            # Check for report (JSON format)
            report_file = results_path / 'report.json'
            if report_file.exists():
                with open(report_file, 'r') as f:
                    results['report'] = json.load(f)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded image"""
    if diagnoser is None:
        return jsonify({'error': 'Model not loaded. Train first with: python main.py'}), 400
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temporarily
        temp_path = 'temp_predict.jpg'
        file.save(temp_path)
        
        # Make prediction
        result = diagnoser.predict_single_image(temp_path)
        
        # Load image for preview
        with open(temp_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': float(result['confidence']),
            'raw_score': float(result['raw_score']),
            'image': f'data:image/jpeg;base64,{img_data}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/info')
def info():
    """Get system information"""
    try:
        import tensorflow as tf
        import cv2
        import numpy
        
        return jsonify({
            'tensorflow_version': tf.__version__,
            'opencv_version': cv2.__version__,
            'numpy_version': numpy.__version__,
            'model_exists': (models_path / 'best_model.h5').exists(),
            'model_loaded': diagnoser is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def open_browser():
    """Open browser after short delay"""
    time.sleep(2)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Initialize app
    init_app()
    
    # Create templates directory if it doesn't exist
    Path('templates').mkdir(exist_ok=True)
    
    # Start browser in background thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    print("\n" + "="*70)
    print("MALARIA DETECTION SYSTEM - WEB DASHBOARD")
    print("="*70)
    print("\n[OK] Starting web server...")
    print("[OK] Opening browser in 2 seconds...")
    print("\nWeb Dashboard URL: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(debug=False, port=5000, use_reloader=False)
