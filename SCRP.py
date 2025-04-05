import cv2
import mediapipe as mp
from statistics import mode
from collections import deque

# Load the best saved model
model = joblib.load("best_hand_gesture_model_dropZ_full.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pred_window = deque(maxlen=10)  # Sliding window (2 ptrs deque ds) for stabilization

while 1:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    gesture_pred = "No Gesture"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Extract (x,y) coordinates from the 21 landmarks
            coords = []
            for i in range(21):
                coords.extend([hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y])
            sample = np.array(coords).reshape(21, 2)
            # Preprocess: recenter using the 1st landmark and normalize using the 12th landmark
            wrist = sample[0, :]
            sample = sample - wrist
            mid_tip = sample[11, :]
            scale = np.linalg.norm(mid_tip)
            if scale == 0:
                scale = 1.0
            sample = sample / scale
            sample_flat = sample.flatten().reshape(1, -1)
            
            pred = model.predict(sample_flat)[0]
            pred_window.append(pred)
            # Decoding numeric vals
            gesture_numeric = mode(pred_window)
            gesture_pred = le.inverse_transform([gesture_numeric])[0]
    
    cv2.putText(frame, f"Gesture: {gesture_pred}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Real-Time Gesture Recognition (Drop z)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()