import os
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# =======================================
# Lazy-loaded globals
# =======================================
_reader = None
_model_toddler = None
_model_child = None
_model_adolescent = None

base_path = os.path.dirname(os.path.abspath(__file__))

def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'])
    return _reader

def get_models():
    global _model_toddler, _model_child, _model_adolescent
    if _model_toddler is None:
        from tensorflow.keras.models import load_model
        _model_toddler = load_model(os.path.join(base_path, "QCHAT10", "toddler_model.h5"))
        _model_child = load_model(os.path.join(base_path, "AQ10_Child", "child_model.h5"))
        _model_adolescent = load_model(os.path.join(base_path, "AQ10_Adolescent", "adolescent_model.h5"))
    return _model_toddler, _model_child, _model_adolescent

# =======================================
# Test Route
# =======================================
@app.route('/test-load', methods=['GET'])
def test_load():
    try:
        print("Testing model load...")

        import tensorflow as tf
        from tensorflow.keras.models import load_model

        model_path = os.path.join(
            base_path,
            "QCHAT10",
            "toddler_model.h5"
        )

        print(f"TensorFlow version: {tf.__version__}")
        print(f"Model path: {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")

        model = load_model(model_path)

        print("✅ Model loaded successfully!")

        return jsonify({
            'status': 'ok',
            'tf_version': tf.__version__
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())

        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
# =======================================
# Prediction Route
# =======================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("🔄 Starting prediction...")
        model_toddler, model_child, model_adolescent = get_models()
        print("✅ Models loaded")
        
        data = request.get_json()
        print("📥 Received data:", data)
        
        age = float(data.get('Age', 0))
        print(f"Age: {age}")
        
        if age > 15:
            return jsonify({'error': 'Screening available only for under 16'}), 400
        elif age < 1:
            return jsonify({'error': 'Screening available only for children above 1'}), 400
        
        features = [data.get(f'A{i}') for i in range(1, 11)]
        print(f"Features: {features}")
        
        if None in features:
            return jsonify({'error': 'Missing one or more question values (A1–A10)'}), 400
        
        features.append(age)
        X_input = np.array(features).reshape(1, -1)
        print(f"X_input shape: {X_input.shape}")
        
        if age < 4:
            model, model_name = model_toddler, "Toddler (QCHAT-10)"
        elif age < 12:
            model, model_name = model_child, "Child (AQ-10 Child)"
        else:
            model, model_name = model_adolescent, "Adolescent (AQ-10 Adolescent)"
        
        print(f"Using model: {model_name}")
        prediction = model.predict(X_input)
        print(f"Prediction: {prediction}")
        
        pred_label = 1 if prediction[0][0] >= 0.5 else 0
        result_text = "YES" if pred_label == 1 else "NO"
        
        return jsonify({
            'model_used': model_name,
            'age': age,
            'prediction_score': float(prediction[0][0]),
            'result': result_text
        })
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# =======================================
# Health Check Route
# =======================================
@app.route('/')
def index():
    return jsonify({'message': 'Autism Screening API is running 🚀'})


# =======================================
# PMDC Verification Route
# =======================================
@app.route("/extract", methods=["POST"])
 def extract():
    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["file"]
    image_bytes = file.read()

   from pmdc import run_ocr, extract_pmdc_data
    reader = get_reader()
    ocr_results = run_ocr(image_bytes)
    data = extract_pmdc_data(ocr_results)

    return data


# =======================================
# Run Flask App
# =======================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
