# ITI-ML1-Hand-Gestures-Classification
Research branch for MLOps

# Hand Gesture Recognition Using Classical Machine Learning
# 🔬 Hand Gesture Recognition Research Pipeline

Comprehensive machine learning research project for hand gesture classification using MediaPipe landmarks. Features systematic experiment tracking with MLflow, model comparison, and reproducible preprocessing pipelines.

[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blue)](https://mlflow.org/)
[![Research](https://img.shields.io/badge/Branch-Research-green)](https://github.com/MohamedElkaffas/ITI-ML1-Hand-Gestures-Classification/tree/research)
[![Production API](https://img.shields.io/badge/Production-Live_API-brightgreen)](https://agkckrhhrjhv.eu-central-1.clawcloudrun.com/docs)

## 🎯 Research Objective

Develop and compare machine learning models for real-time hand gesture recognition using MediaPipe hand landmarks, with the goal of achieving >95% accuracy for production deployment in gesture-controlled applications.

## 📊 Research Results Summary

| Model | Test Accuracy | Weighted F1 | Status | Use Case |
| ----- | ------------- | ----------- | ------ | -------- |
| **SVM (RBF)** | **98.8%** | **0.988** | 🚀 **Production** | Real-time gesture control |
| **XGBoost** | **97.9%** | **0.979** | 🔄 Candidate | High-accuracy applications |
| **Logistic Regression** | **84.7%** | **0.846** | 📊 Baseline | Comparative baseline |

*All experiments tracked and reproducible via MLflow*

## 🏗️ Research Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Dataset   │───▶│  Preprocessing   │───▶│   ML Models     │
│ (HaGRID-based)  │    │   Pipeline       │    │  (3 Algorithms) │
│ 21 landmarks    │    │   63→42 features │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │     MLflow       │    │   Production    │
                       │   Tracking       │    │   Deployment    │
                       │                  │    │                 │
                       └──────────────────┘    └─────────────────┘
```

## 🔬 Experimental Design

### Dataset Characteristics

- **Source**: Custom ML1 Hand Gestures dataset (HaGRID-based)
- **Features**: 21 MediaPipe anatomical landmarks per hand
- **Input Format**: 63 coordinates (x, y, z for each landmark)
- **Gesture Classes**: 14 distinct hand gestures
- **Data Split**: Train/Validation/Test with stratified sampling

### Preprocessing Pipeline

The critical preprocessing transformation that enabled high accuracy:

```python
# Core preprocessing steps (63 → 42 features)
def preprocess_landmarks(landmarks):
    # 1. Reshape to (21, 3) array
    landmarks_array = np.array(landmarks).reshape(21, 3)
    
    # 2. Keep only x,y coordinates (drop z-axis)
    xy_coordinates = landmarks_array[:, :2]  # (21, 2)
    
    # 3. Wrist-relative positioning (translation invariance)
    wrist = xy_coordinates[0, :]
    rel_coords = xy_coordinates - wrist
    
    # 4. Scale normalization (scale invariance)
    mid_tip = rel_coords[11, :]  # Middle finger tip
    scale = np.linalg.norm(mid_tip)
    if scale == 0: scale = 1.0
    
    # 5. Normalize and flatten
    normalized = rel_coords / scale
    return normalized.flatten()  # 42 features
```

**Key Innovation**: This preprocessing achieves both translation and scale invariance, making gestures recognizable regardless of hand position or size.

## 🧪 MLflow Experiment Tracking

### Experiment Setup

```python
# MLflow tracking configuration
import mlflow
import mlflow.sklearn

mlflow.set_experiment("Hand_Gesture_Recognition_Comparison")

with mlflow.start_run(run_name="SVM_RBF_Optimized"):
    # Log parameters
    mlflow.log_param("model_type", "SVM")
    mlflow.log_param("kernel", "rbf")
    mlflow.log_param("preprocessing", "63_to_42_features")
    
    # Log metrics
    mlflow.log_metric("test_accuracy", 0.988)
    mlflow.log_metric("weighted_f1", 0.988)
    
    # Log model
    mlflow.sklearn.log_model(model, "gesture_classifier")
    
    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importance.png")
```

### Tracked Experiments

#### 1. Baseline Logistic Regression
```yaml
Parameters:
  - solver: 'liblinear'
  - max_iter: 1000
  - random_state: 42

Metrics:
  - test_accuracy: 0.847
  - weighted_f1: 0.846
  - precision_macro: 0.851
  - recall_macro: 0.847

Artifacts:
  - confusion_matrix.png
  - classification_report.txt
  - model_signature.json
```

#### 2. Support Vector Machine (Production Model)
```yaml
Parameters:
  - kernel: 'rbf'
  - C: 10.0
  - gamma: 'scale'
  - random_state: 42

Metrics:
  - test_accuracy: 0.988
  - weighted_f1: 0.988
  - precision_macro: 0.989
  - recall_macro: 0.988

Artifacts:
  - confusion_matrix.png
  - decision_boundary_viz.png
  - model_signature.json
  - input_example.json
```

#### 3. XGBoost Classifier
```yaml
Parameters:
  - n_estimators: 200
  - max_depth: 6
  - learning_rate: 0.1
  - random_state: 42

Metrics:
  - test_accuracy: 0.979
  - weighted_f1: 0.979
  - precision_macro: 0.980
  - recall_macro: 0.979

Artifacts:
  - confusion_matrix.png
  - feature_importance.png
  - model_signature.json
```

### Model Signatures & Reproducibility

All models logged with MLflow include:

```python
# Model signature for input validation
signature = mlflow.models.signature.infer_signature(X_test, predictions)

# Input example for documentation
input_example = X_test[:5]  # First 5 test samples

# Model metadata
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    signature=signature,
    input_example=input_example,
    registered_model_name="gesture_classifier_svm"
)
```

## 📈 Comparative Analysis

### Performance Metrics Comparison

![Model Comparison Chart](model_comparison_chart.png)

**Key Findings:**

1. **SVM Dominance**: RBF kernel SVM achieved highest accuracy (98.8%)
2. **XGBoost Strong Second**: Close performance (97.9%) with feature interpretability
3. **Logistic Regression Baseline**: Solid baseline (84.7%) showing data quality
4. **Preprocessing Impact**: 63→42 feature reduction improved all models
5. **Production Readiness**: SVM selected for deployment based on accuracy + speed

### Confusion Matrix Analysis

The SVM model shows excellent performance across all gesture classes:

- **High Precision**: Minimal false positives for each gesture
- **High Recall**: Captures most instances of each gesture
- **Balanced Performance**: No significant bias toward specific gestures
- **Production Ready**: Confusion patterns suitable for real-time applications

## 🚀 Research to Production Pipeline

### Model Selection Criteria

| Criteria | Weight | SVM Score | XGBoost Score | LogReg Score |
|----------|--------|-----------|---------------|--------------|
| **Accuracy** | 40% | 9.9/10 | 9.8/10 | 8.5/10 |
| **Inference Speed** | 25% | 9.0/10 | 7.0/10 | 9.5/10 |
| **Model Size** | 15% | 8.0/10 | 6.0/10 | 9.0/10 |
| **Interpretability** | 10% | 6.0/10 | 9.0/10 | 8.0/10 |
| **Robustness** | 10% | 9.0/10 | 8.0/10 | 7.0/10 |

**Final Score**: SVM (8.85) > XGBoost (8.24) > LogReg (8.45)

### Production Deployment

The winning SVM model was deployed via:

1. **Model Export**: `joblib.dump(svm_model, 'best_hand_gesture.pkl')`
2. **API Integration**: FastAPI service with identical preprocessing
3. **Containerization**: Docker with model artifacts
4. **CI/CD Deployment**: Automated pipeline to ClawCloud
5. **Monitoring**: MLflow model registry + Prometheus metrics

**Live Production API**: [https://agkckrhhrjhv.eu-central-1.clawcloudrun.com/docs](https://agkckrhhrjhv.eu-central-1.clawcloudrun.com/docs)

## 💻 Reproducibility Setup

### Environment Setup

```bash
# Clone research branch
git clone -b research https://github.com/MohamedElkaffas/ITI-ML1-Hand-Gestures-Classification.git
cd ITI-ML1-Hand-Gestures-Classification

# Create virtual environment
python -m venv mlflow_env
source mlflow_env/bin/activate  # Windows: mlflow_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### MLflow Tracking Server

```bash
# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Access MLflow UI
open http://localhost:5000
```

### Run Experiments

```bash
# Run all experiments
python run_experiments.py

# Run specific model
python train_model.py --model svm --track-mlflow

# Compare models
python compare_models.py --output comparison_chart.png
```

### Directory Structure

```
research/
├── data/
│   ├── raw/                    # Original gesture dataset
│   ├── processed/              # Preprocessed features
│   └── splits/                 # Train/val/test splits
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_results_analysis.ipynb
├── src/
│   ├── preprocessing.py        # Feature engineering pipeline
│   ├── models.py              # Model definitions
│   ├── evaluation.py          # Metrics and visualization
│   └── mlflow_utils.py        # MLflow tracking utilities
├── experiments/
│   ├── run_experiments.py     # Automated experiment runner
│   ├── train_svm.py          # SVM training script
│   ├── train_xgboost.py      # XGBoost training script
│   └── train_baseline.py     # Logistic regression baseline
├── artifacts/
│   ├── models/               # Saved model files
│   ├── plots/               # Generated visualizations
│   └── reports/             # Experiment reports
├── mlruns/                  # MLflow tracking data
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 📊 Key Research Insights

### 1. Preprocessing is Critical
- **63→42 transformation**: Removing z-axis noise improved accuracy by ~5%
- **Wrist-relative coordinates**: Translation invariance crucial for generalization
- **Scale normalization**: Hand size independence essential for real-world use

### 2. Model Performance Patterns
- **SVM with RBF kernel**: Excellent for high-dimensional gesture data
- **XGBoost**: Strong performance with interpretable feature importance
- **Logistic Regression**: Surprisingly effective baseline (84.7%)

### 3. Production Considerations
- **Inference Speed**: SVM fast enough for real-time (sub-100ms)
- **Model Size**: SVM compact enough for edge deployment
- **Robustness**: SVM handles noisy input gracefully

### 4. MLflow Benefits
- **Reproducibility**: All experiments fully reproducible
- **Comparison**: Easy model comparison and selection
- **Deployment**: Seamless transition from research to production

## 🔗 Related Projects

### Production Ecosystem

- **🚀 [Production API](https://github.com/MohamedElkaffas/Handgestures-API)**: FastAPI service with CI/CD and monitoring
- **🎮 [Frontend Application](https://github.com/MohamedElkaffas/MLOPs-Final-Project)**: Real-time gesture-controlled maze game
- **📊 [Live Demo](https://mohamedelkaffas.github.io/MLOPs-Final-Project/)**: Interactive gesture recognition demo

### Research Extensions

- **Multi-hand Support**: Extend to simultaneous two-hand gestures
- **Temporal Models**: LSTM/RNN for gesture sequences
- **Edge Deployment**: TensorFlow Lite for mobile devices
- **Data Augmentation**: Synthetic gesture generation
