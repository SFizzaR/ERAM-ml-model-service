# ERAM-ml-model-service

A Flask-based Machine Learning API for autism screening across multiple age groups using questionnaire-based screening tools and neural network models.

## Overview

This project provides autism screening predictions for:

- Toddlers (QCHAT-10)
- Children (AQ-10 Child)
- Adolescents (AQ-10 Adolescent)

The API automatically selects the appropriate machine learning model based on the user's age and returns a screening prediction score indicating the likelihood of Autism Spectrum Disorder (ASD) traits.

The project demonstrates the complete machine learning lifecycle:

- Data preprocessing
- Model training and evaluation
- TensorFlow/Keras deployment
- REST API development with Flask
- Cloud deployment considerations


## Features

### Multi-Age Autism Screening

The system supports screening across three age groups:

| Age Range | Model Used |
|------------|------------|
| 1 – 3 years | QCHAT-10 Toddler |
| 4 – 11 years | AQ-10 Child |
| 12 – 15 years | AQ-10 Adolescent |

The API automatically selects the correct model based on the submitted age.


### Machine Learning Inference

- TensorFlow/Keras neural network models
- Real-time prediction API
- Lazy-loaded model architecture
- Age-based model routing
- JSON-based request/response format


### REST API

Available endpoints:

| Endpoint | Method | Purpose |
|-----------|---------|---------|
| `/` | GET | Health check |
| `/test-load` | GET | Verify model loading |
| `/predict` | POST | Generate screening prediction |


## Technology Stack

### Backend

- Python
- Flask
- Gunicorn

### Machine Learning

- TensorFlow 2.13
- Keras
- NumPy
- Scikit-Learn

### Computer Vision / OCR

- OpenCV
- EasyOCR (Local Version)


## Model Training

Three separate neural network models were trained for different age groups.

### Age Groups

| Model | Age Range | Questionnaire |
|---------|------------|--------------|
| Toddler | 1 – 3 years | QCHAT-10 |
| Child | 4 – 11 years | AQ-10 Child |
| Adolescent | 12 – 15 years | AQ-10 Adolescent |

### Training Pipeline

The training workflow included:

1. Dataset preprocessing
2. Feature extraction
3. Label generation from screening responses
4. Train-test split (80/20)
5. Neural network training using TensorFlow/Keras
6. Early stopping to prevent overfitting
7. Best-model checkpointing
8. Model evaluation using classification metrics

### Neural Network Architecture

```python
model = keras.Sequential([
    layers.Input(shape=(11,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

### Training Configuration

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Batch Size: 16
- Maximum Epochs: 100
- Validation Split: 20%
- Early Stopping Patience: 5

### Saved Models

```text
toddler_model.h5
child_model.h5
adolescent_model.h5
```

Models were exported in `.h5` format to ensure compatibility with TensorFlow 2.13 deployment environments.

## API Endpoints

### Health Check

```http
GET /
```

Response:

```json
{
  "message": "Autism Screening API is running 🚀"
}
```


### Model Load Verification

```http
GET /test-load
```

Used to verify that the TensorFlow models load correctly.

Response:

```json
{
  "status": "ok",
  "tf_version": "2.13.0"
}
```


### Autism Screening Prediction

```http
POST /predict
```

Request:

```json
{
  "Age": 5,
  "A1": 1,
  "A2": 0,
  "A3": 1,
  "A4": 0,
  "A5": 1,
  "A6": 0,
  "A7": 1,
  "A8": 0,
  "A9": 1,
  "A10": 0
}
```

Response:

```json
{
  "model_used": "Child (AQ-10 Child)",
  "age": 5,
  "prediction_score": 0.84,
  "result": "YES"
}
```


## Running Locally

### Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Application

```bash
python app.py
```

Server will start on:

```text
http://localhost:5000
```

## Example Request

Using curl:

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "Age": 5,
  "A1": 1,
  "A2": 0,
  "A3": 1,
  "A4": 0,
  "A5": 1,
  "A6": 0,
  "A7": 1,
  "A8": 0,
  "A9": 1,
  "A10": 0
}'
```


## PMDC Verification Module

The original version of this project included a PMDC document verification workflow using OCR-based text extraction.

The implementation remains available in the codebase through:

```text
pmdc.py
```

and can be enabled locally by uncommenting the `/extract` endpoint in `app.py`.

This functionality was preserved for local development and experimentation.

## Deployment Notes

The deployed version focuses on the autism screening API.

The PMDC OCR workflow relies on EasyOCR, which introduces PyTorch dependencies that significantly increase deployment size and complexity.

During deployment, dependency resolution attempted to install GPU/CUDA-related PyTorch packages, causing excessive build times and deployment failures on the selected hosting platform.

As a result:

- Autism screening endpoints remain fully functional.
- OCR functionality remains available in the source code.
- OCR can be enabled for local execution.
- The deployed version excludes OCR-related functionality.

This was an engineering tradeoff made to ensure a stable and lightweight deployment environment.


## What This Project Demonstrates

- Machine Learning Model Training
- TensorFlow/Keras Inference Pipelines
- Neural Network Classification
- Data Preprocessing
- REST API Development
- Flask Backend Development
- Healthcare Screening Applications
- Cloud Deployment Troubleshooting
- Dependency Management for ML Systems
- Production Model Loading Strategies


## Limitations

- This application performs screening predictions only.
- It does not provide a medical diagnosis.
- Clinical assessment by qualified professionals is required for diagnosis.
- OCR functionality may require additional local dependencies.
- Predictions should be used for educational and research purposes only.

## Future Improvements

- Dockerized deployment
- CPU-optimized OCR deployment
- CI/CD integration


## Disclaimer

This project is intended solely for educational, research, and demonstration purposes.

The predictions generated by the system are based on questionnaire responses and machine learning models. They should not be interpreted as a medical diagnosis or used as a substitute for professional clinical evaluation.
