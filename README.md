# ITI-ML1-Hand-Gestures-Classification
A project for the ITI course ML1

# Hand Gesture Recognition Using Classical Machine Learning

This project implements a complete workflow for hand gesture recognition using hand landmark data extracted by MediaPipe from the HaGRID dataset. Each sample in the dataset contains 21 hand landmarks (with x, y, z coordinates) along with a corresponding gesture label. In our workflow, we drop the z coordinate, recenter and normalize the remaining features, and then train several classical machine learning models. The models are compared and the best is selected based on the **weighted F1 score**, which is particularly suited for imbalanced multi-class data.

[Watch the demo video](https://drive.google.com/file/d/1SakmyPTurVdVCT1fUeCFUF39hZww_tL3/view?usp=sharing)


## Project Overview

### 1. Data Loading & Analysis
- **Input Data:**  
  The dataset (`hand_landmarks_data.csv`) contains 63 feature columns (representing 21 landmarks with x1, y1, z1, …, x21, y21, z21) plus a `label` column.
- **Analysis:**  
  Descriptive statistics are computed, and distributions of key features (e.g., `x1` and `y1`) are plotted to understand data spread.

### 2. Data Visualization
- **Raw Landmarks:**  
  Raw (x, y) coordinates from a few random samples are visualized to verify the quality and orientation of the landmark data.

### 3. Data Preprocessing
- **Dropping z Coordinate:**  
  Only the x and y values are retained, resulting in 42 features per sample.
- **Recentering and Normalization:**  
  The landmarks are recentered by subtracting the first landmark (assumed to be the wrist) and normalized by dividing by the Euclidean distance from the wrist to the 12th landmark (assumed to be the mid-finger tip).
- **Label Encoding:**  
  Gesture labels (e.g., "call", "fist", etc.) are converted into numeric values using `LabelEncoder` for model compatibility. These numeric labels are later decoded back to the original labels for evaluation and real-time inference.

### 4. Data Splitting
The processed dataset is split into:
- **Training Set:** 60%
- **Validation Set:** 20%
- **Test Set:** 20%

This split ensures a robust evaluation of model performance and helps detect overfitting.

### 5. Model Training, Validation & Comparison
Multiple classifiers are tuned using GridSearchCV (with 5-fold cross-validation) on the training set. The models evaluated include:

- **Random Forest:**  
  An ensemble of decision trees that averages multiple predictions to reduce variance. It generally performs well on complex data.
  
- **SVM (Support Vector Machine):**  
  A powerful algorithm that finds the optimal hyperplane for classification. It is particularly effective when classes are well-separated.
  
- **Logistic Regression:**  
  A simple yet effective linear model used as a strong baseline for classification. It is computationally efficient and interpretable.
  
- **Decision Tree:**  
  A tree-based method that splits the data based on feature thresholds. It is easy to interpret but can overfit without pruning.
  
- **AdaBoost:**  
  A boosting algorithm that combines many weak learners (typically decision stumps). Because it uses simple base learners, it may underperform on complex data.
  
- **XGBoost:**  
  An advanced gradient boosting framework that uses both first- and second-order gradients to optimize the loss function. It is known for its high performance and scalability.
  
- **K-Nearest Neighbors (KNN):**  
  A non-parametric method that classifies samples based on the majority class among their nearest neighbors. It can be computationally intensive with large datasets.

**Note on AdaBoost:**  
AdaBoost often uses very simple (weak) learners, which might not capture the complexity of the hand gesture data. As a result, its performance in our experiments was significantly poorer compared to other models.

Each model’s performance is evaluated using:
- **Accuracy**
- **Weighted Precision**
- **Weighted Recall**
- **Weighted F1 Score**

For imbalanced multi-class data, **weighted F1 score** is the primary metric because it balances precision and recall while taking the class distribution into account.

The best model—based on the highest weighted F1 score—is saved as `best_hand_gesture_model_dropZ_full.pkl`. A ranking table and learning curve are also generated for further analysis.

### 6. Real-Time Inference
A separate Python script (`SCRPT.py`) performs real-time gesture recognition using:
- **MediaPipe:** To extract hand landmarks from a live video feed.
- **Preprocessing:** The same as during training (dropping z, recentering, normalizing).
- **Prediction Stabilization:** A sliding window (mode) is used to smooth out predictions.
- **Label Decoding:** The numeric predictions are converted back to the original gesture names.

Conclusions:-

- The dataset was preprocessed by dropping the z coordinate, recentering the (x, y) values, and normalizing them based on the mid-finger tip distance.

Gesture labels were encoded to numeric values using LabelEncoder, and later decoded for reporting and inference.

The data was split into training (60%), validation (20%), and test (20%) sets to ensure robust model evaluation and overfitting checks.

Multiple models were tuned using GridSearchCV with 5-fold cross-validation.

Weighted F1 score was chosen as the primary metric because it accounts for both precision and recall while adjusting for class imbalance.

AdaBoost, which relies on weak learners, performed poorly relative to more complex models.

The best model was selected based on the weighted F1 score and is used in a real-time inference script that leverages MediaPipe and OpenCV.

**To run the real-time inference:**
1. Ensure `best_hand_gesture_model_dropZ_full.pkl` are in your project directory.
2. Execute:
   ```bash
   python SCRPT.py
