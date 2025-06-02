"""
Training script that loads existing SVM and trains 2 new baseline models.
Logs all models to MLflow for comparison.
"""

import os
import pickle
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from preprocess import process_hand_landmarks_xy

def log_model_to_mlflow(model, model_name, X_sample, y_sample, params: dict, metrics: dict):
    """
    Log model to MLflow with consistent format.
    """
    signature = infer_signature(X_sample, model.predict(X_sample))
    
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics, step=0)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_sample.iloc[:3].to_dict(orient="records")
        )
        
        print(f"Logged {model_name}: test_acc={metrics['test_accuracy']:.4f}, test_f1={metrics['test_f1']:.4f}")

def main():
    
    CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hand_landmarks_data.csv")    df = pd.read_csv(CSV_PATH)
    print(f"Loaded data: {df.shape}")
    
    X_raw = df.drop(columns=["label"])
    y_raw = df["label"]
    
    X_processed = X_raw.apply(process_hand_landmarks_xy, axis=1, result_type="expand")
    X_processed = pd.DataFrame(X_processed)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    le_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"..", "models", "label_encoder.pkl")
    
    with open(le_path, "wb") as f_le:
        pickle.dump(le, f_le)
    print(f"Saved LabelEncoder to {le_path}")
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    print(f"Data splits: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Hand_Gesture_Comparison")
    
    print("Training Baseline (LogisticRegression)...")
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
    
    log_model_to_mlflow(lr, "Baseline_LogReg", X_train, y_train, lr_params, metrics_A)
    
    svm_pickle_path = os.path.join(os.path.dirname(__file__),"..", "models", "best_hand_gesture.pkl")    
    if os.path.exists(svm_pickle_path):
        print("Evaluating Loaded SVM...")
        with open(svm_pickle_path, "rb") as f_svm:
            loaded_svm = pickle.load(f_svm)
        
        y_val_pred_S = loaded_svm.predict(X_val)
        y_test_pred_S = loaded_svm.predict(X_test)
        
        metrics_S = {
            "val_accuracy": accuracy_score(y_val, y_val_pred_S),
            "val_f1": f1_score(y_val, y_val_pred_S, average="weighted"),
            "test_accuracy": accuracy_score(y_test, y_test_pred_S),
            "test_f1": f1_score(y_test, y_test_pred_S, average="weighted"),
        }
        
        svm_params = {"C": 10, "kernel": "rbf", "probability": True}
        log_model_to_mlflow(loaded_svm, "Loaded_SVM", X_train, y_train, svm_params, metrics_S)
    else:
        print(f"SVM not found at {svm_pickle_path}")
    
    print("Training Improved (XGBoost)...")
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
    
    log_model_to_mlflow(xgb, "Improved_XGB", X_train, y_train, xgb_params, metrics_B)
    
    all_metrics = [metrics_A, metrics_S, metrics_B]
    model_names = ["Baseline_LogReg", "Loaded_SVM", "Improved_XGB"]
    models = [lr, loaded_svm, xgb]
    
    best_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]["test_f1"])
    best_model = models[best_idx]
    best_name = model_names[best_idx]
    
    best_model_path = os.path.join(os.path.dirname(__file__), "models/best_hand_gesture.pkl")
    with open(best_model_path, "wb") as f_best:
        pickle.dump(best_model, f_best)
    
    print(f"Best model: {best_name} (F1: {all_metrics[best_idx]['test_f1']:.4f})")
    print(f"Saved to {best_model_path}")
    
    test_accs = [m["test_accuracy"] for m in all_metrics]
    test_f1s = [m["test_f1"] for m in all_metrics]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(model_names))
    plt.bar(x - 0.2, test_accs, 0.4, label="Test Accuracy", alpha=0.8)
    plt.bar(x + 0.2, test_f1s, 0.4, label="Test F1", alpha=0.8)
    plt.xticks(x, model_names, rotation=15)
    plt.ylabel("Score")
    plt.title("Model Comparison (Test Set Performance)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.join(os.path.dirname(__file__), "reports"), exist_ok=True)
    chart_path = os.path.join(os.path.dirname(__file__), "reports/model_comparison.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison chart to {chart_path}")
    print("All models trained and logged to MLflow!")
    print("Run 'mlflow ui' to view the experiment dashboard")

if __name__ == "__main__":
    main()