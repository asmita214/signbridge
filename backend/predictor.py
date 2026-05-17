import pickle
import numpy as np
from tensorflow import keras
import mediapipe as mp

print("Loading models...")

with open("models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le_static = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/le_dynamic.pkl", "rb") as f:
    le_dynamic = pickle.load(f)

# Build NN architecture and load weights
nn_model = keras.Sequential([
    keras.layers.Input(shape=(63,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(28, activation='softmax')
])
nn_model.load_weights("models/nn_weights.weights.h5")

# Build LSTM architecture and load weights
lstm_model = keras.Sequential([
    keras.layers.Input(shape=(30, 63)),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dropout(0.3),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.Dropout(0.3),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(20, activation='softmax')
])
lstm_model.load_weights("models/lstm_weights.weights.h5")

print("All models loaded successfully")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def extract_landmarks(frame_rgb):
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        coords = []
        for point in lm.landmark:
            coords.extend([point.x, point.y, point.z])
        return np.array(coords)
    return None

def predict_static(landmarks):
    rf_proba = rf_model.predict_proba([landmarks])[0]
    landmarks_scaled = scaler.transform([landmarks])
    nn_proba = nn_model.predict(landmarks_scaled, verbose=0)[0]
    ensemble_proba = (rf_proba * 0.4) + (nn_proba * 0.6)
    predicted_index = np.argmax(ensemble_proba)
    confidence = ensemble_proba[predicted_index]
    predicted_label = le_static.inverse_transform([predicted_index])[0]
    return predicted_label, float(confidence)

def predict_dynamic(sequence):
    sequence = np.array(sequence)
    input_data = sequence.reshape(1, 30, 63)
    proba = lstm_model.predict(input_data, verbose=0)[0]
    predicted_index = np.argmax(proba)
    confidence = proba[predicted_index]
    predicted_label = le_dynamic.inverse_transform([predicted_index])[0]
    return predicted_label, float(confidence)