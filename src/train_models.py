"""
Training script that loads existing SVM and trains 2 new baseline models.
Logs all models to MLflow for comparison with enhanced metadata and artifacts.
"""

import os
import pickle
import joblib 
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import process_hand_landmarks_xy

def log_model_to_mlflow_enhanced(model, model_name, X_train, X_test, y_train, y_test, 
                                params: dict, metrics: dict, label_encoder, df):
    """
    Enhanced MLflow logging with signature, metadata, encoders, and game mappings.
    """
    signature = infer_signature(X_test, model.predict(X_test))
    
    with mlflow.start_run(run_name=model_name):
        # LOG PARAMETERS
        mlflow.log_params(params)
        
        # LOG ENHANCED METRICS
        enhanced_metrics = {
            **metrics,
            "num_classes": len(label_encoder.classes_),
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": X_test.shape[1]
        }
        mlflow.log_metrics(enhanced_metrics)
        
        # LOG MODEL WITH SIGNATURE & INPUT EXAMPLE
        input_example = X_test.iloc[:3] if hasattr(X_test, 'iloc') else X_test[:3]
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            metadata={
                "model_type": type(model).__name__,
                "training_framework": "scikit-learn",
                "use_case": "hand_gesture_maze_control",
                "created_for": "real_time_gesture_recognition"
            }
        )
        
        # LOG DATASET METADATA
        dataset_info = {
            "total_samples": len(df),
            "features": X_test.shape[1],
            "gesture_classes": label_encoder.classes_.tolist(),
            "class_distribution": df['label'].value_counts().to_dict(),
            "feature_names": [f"landmark_{i}" for i in range(X_test.shape[1])],
            "data_source": "MediaPipe_hand_landmarks",
            "coordinate_system": "normalized_xy_coordinates",
            "preprocessing": "process_hand_landmarks_xy"
        }
        mlflow.log_dict(dataset_info, "dataset_metadata.json")
        
        # LOG GESTURE-TO-MOVEMENT MAPPING (For maze game)
        gesture_mapping = {
            "gesture_to_number": {gesture: int(num) for num, gesture in enumerate(label_encoder.classes_)},
            "number_to_gesture": {int(num): gesture for num, gesture in enumerate(label_encoder.classes_)},
            "maze_game_controls": {
                "thumbs_up": "UP",
                "like": "UP", 
                "thumbs_down": "DOWN",
                "dislike": "DOWN",
                "fist": "LEFT",
                "peace": "RIGHT",
                "stop": "STOP",
                "palm": "STOP",
                "ok": "PAUSE"
            },
            "available_gestures": label_encoder.classes_.tolist()
        }
        mlflow.log_dict(gesture_mapping, "gesture_mappings.json")
        
        # LOG CLASSIFICATION REPORT
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, 
                                     target_names=label_encoder.classes_, 
                                     output_dict=True)
        mlflow.log_dict(report, "classification_report.json")
        
        # LOG CONFUSION MATRIX
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, 
                           index=label_encoder.classes_, 
                           columns=label_encoder.classes_)
        
        # Save and log confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(cm_path)
        plt.close()
        
        # Clean up local file
        if os.path.exists(cm_path):
            os.remove(cm_path)
        
        # LOG LABEL ENCODER (CRITICAL for later inference)
        le_temp_path = "temp_label_encoder.pkl"
        with open(le_temp_path, "wb") as f:
            pickle.dump(label_encoder, f)
        mlflow.log_artifact(le_temp_path, "preprocessors/label_encoder.pkl")
        os.remove(le_temp_path)  # Clean up
        
        print(f"Enhanced logging for {model_name}:")
        print(f"Metrics: {enhanced_metrics['test_accuracy']:.4f} acc, {enhanced_metrics['test_f1']:.4f} f1")
        print(f"Gestures: {len(label_encoder.classes_)} classes")
        print(f"Artifacts: model + metadata + encoders + mappings")

def evaluate_model(model, X_test, y_test):
    """Helper function to evaluate model performance."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return acc, f1

def main():
    
    CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hand_landmarks_data.csv")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded data: {df.shape}")
    
    X_raw = df.drop(columns=["label"])
    y_raw = df["label"]
    
    X_processed = X_raw.apply(process_hand_landmarks_xy, axis=1, result_type="expand")
    X_processed = pd.DataFrame(X_processed)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    le_path = os.path.join(os.path.dirname(__file__), "..", "models", "label_encoder.pkl")
    with open(le_path, "wb") as f_le:
        pickle.dump(le, f_le)
    print(f"Saved LabelEncoder to {le_path}")
    print(f"Available gestures: {le.classes_}")
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    print(f"Data splits: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Hand_Gesture_Maze_Controller")
    
    # Store models for comparison
    models_to_compare = []
    
    # BASELINE: LOGISTIC REGRESSION 
    print("\nTraining Baseline (LogisticRegression)...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_params = {"C": 1.0, "solver": "liblinear", "max_iter": 1000}
    
    lr.fit(X_train, y_train)
    
    y_val_pred_A = lr.predict(X_val)
    y_test_pred_A = lr.predict(X_test)
    
    metrics_A = {
        "val_accuracy": accuracy_score(y_val, y_val_pred_A),
        "val_f1": f1_score(y_val, y_val_pred_A, average="weighted"),
        "test_accuracy": accuracy_score(y_test, y_test_pred_A),
        "test_f1": f1_score(y_test, y_test_pred_A, average="weighted"),
    }
    
    log_model_to_mlflow_enhanced(lr, "Baseline_LogReg", X_train, X_test, y_train, y_test, 
                               lr_params, metrics_A, le, df)
    models_to_compare.append(("Baseline_LogReg", lr, metrics_A["test_accuracy"], metrics_A["test_f1"]))
    
    # EXISTING SVM MODEL 
    svm_pickle_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_hand_gesture.pkl")    
    if os.path.exists(svm_pickle_path):
        print("\nEvaluating Loaded SVM...")
        # Fixed: Load with joblib directly, not through file handle
        loaded_svm = joblib.load(svm_pickle_path)
        
        y_val_pred_S = loaded_svm.predict(X_val)
        y_test_pred_S = loaded_svm.predict(X_test)
        
        metrics_S = {
            "val_accuracy": accuracy_score(y_val, y_val_pred_S),
            "val_f1": f1_score(y_val, y_val_pred_S, average="weighted"),
            "test_accuracy": accuracy_score(y_test, y_test_pred_S),
            "test_f1": f1_score(y_test, y_test_pred_S, average="weighted"),
        }
        
        svm_params = {"C": 10, "kernel": "rbf", "probability": True, "model_source": "pre_trained"}
        log_model_to_mlflow_enhanced(loaded_svm, "Loaded_SVM", X_train, X_test, y_train, y_test,
                                   svm_params, metrics_S, le, df)
        models_to_compare.append(("Loaded_SVM", loaded_svm, metrics_S["test_accuracy"], metrics_S["test_f1"]))
    else:
        print(f"⚠️  SVM not found at {svm_pickle_path}")
    
    # IMPROVED: XGBOOST
    print("\nTraining Improved (XGBoost)...")
    xgb = XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softprob'
    )
    xgb_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "objective": "multi:softprob"}
    
    xgb.fit(X_train, y_train)
    
    y_val_pred_B = xgb.predict(X_val)
    y_test_pred_B = xgb.predict(X_test)
    
    metrics_B = {
        "val_accuracy": accuracy_score(y_val, y_val_pred_B),
        "val_f1": f1_score(y_val, y_val_pred_B, average="weighted"),
        "test_accuracy": accuracy_score(y_test, y_test_pred_B),
        "test_f1": f1_score(y_test, y_test_pred_B, average="weighted"),
    }
    
    log_model_to_mlflow_enhanced(xgb, "Improved_XGB", X_train, X_test, y_train, y_test,
                               xgb_params, metrics_B, le, df)
    models_to_compare.append(("Improved_XGB", xgb, metrics_B["test_accuracy"], metrics_B["test_f1"]))
    
    # === DETERMINE BEST MODEL ===
    print("\n📊 Model Comparison:")
    for name, model, acc, f1 in models_to_compare:
        print(f"   {name}: Accuracy={acc:.4f}, F1={f1:.4f}")
    
    # Select best model by F1 score
    best_name, best_model, best_acc, best_f1 = max(models_to_compare, key=lambda x: x[3])
    
    # Save best model locally (for compatibility)
    best_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_hand_gesture.pkl")
    joblib.dump(best_model, best_model_path)  # Use joblib for consistency
    
    print(f"\nBest model: {best_name} (F1: {best_f1:.4f})")
    print(f"Saved to {best_model_path}")
    
    # VISUALS
    model_names = [x[0] for x in models_to_compare]
    test_accs = [x[2] for x in models_to_compare]
    test_f1s = [x[3] for x in models_to_compare]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, test_accs, width, label="Test Accuracy", alpha=0.8, color='skyblue')
    plt.bar(x + width/2, test_f1s, width, label="Test F1", alpha=0.8, color='lightcoral')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Hand Gesture Model Comparison (Test Set Performance)')
    plt.xticks(x, model_names, rotation=15)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (acc, f1) in enumerate(zip(test_accs, test_f1s)):
        plt.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save chart
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "reports"), exist_ok=True)
    chart_path = os.path.join(os.path.dirname(__file__), "..", "reports", "model_comparison.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison chart to {chart_path}")
    print("\nMAZE GAME READY! AND MAPPINGS DONE")
    print("Models trained and logged with full metadata")
    print("LabelEncoder saved in MLflow artifacts")
    print("Ready for real-time gesture recognition")
    print("\nRun 'mlflow ui' to view the experiment dashboard")

if __name__ == "__main__":
    main()