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
        _model_toddler = load_model(os.path.join(base_path, "QCHAT10", "toddler_model.keras"))
        _model_child = load_model(os.path.join(base_path, "AQ10_Child", "child_model.keras"))
        _model_adolescent = load_model(os.path.join(base_path, "AQ10_Adolescent", "adolescent_model.keras"))
    return _model_toddler, _model_child, _model_adolescent

# =======================================
# Prediction Route
# =======================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_toddler, model_child, model_adolescent = get_models()
        data = request.get_json()
        print("Received data for prediction:", data)

        age = float(data.get('Age', 0))
        if age > 15:
            return jsonify({'error': 'Screening available only for under 16'}), 400
        elif age < 1:
            return jsonify({'error': 'Screening available only for children above 1'}), 400

        features = [data.get(f'A{i}') for i in range(1, 11)]
        if None in features:
            return jsonify({'error': 'Missing one or more question values (A1–A10)'}), 400

        features.append(age)
        X_input = np.array(features).reshape(1, -1)

        if age < 4:
            model, model_name = model_toddler, "Toddler (QCHAT-10)"
        elif age < 12:
            model, model_name = model_child, "Child (AQ-10 Child)"
        else:
            model, model_name = model_adolescent, "Adolescent (AQ-10 Adolescent)"

        prediction = model.predict(X_input)
        pred_label = 1 if prediction[0][0] >= 0.5 else 0
        result_text = "YES" if pred_label == 1 else "NO"

        return jsonify({
            'model_used': model_name,
            'age': age,
            'prediction_score': float(prediction[0][0]),
            'result': result_text
        })

    except Exception as e:
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