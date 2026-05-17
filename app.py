import cv2
import numpy as np
import pickle
import time
import threading
import mediapipe as mp
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

print("Loading models...")
with open("models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("models/label_encoder.pkl", "rb") as f:
    le_static = pickle.load(f)
with open("models/le_dynamic.pkl", "rb") as f:
    le_dynamic = pickle.load(f)

# LSTM only
from tensorflow import keras
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
print("All models loaded.")
print("Static classes:", le_static.classes_)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-1.5-flash")

def correct_grammar(text):
    try:
        prompt = f"Convert this sign language output to natural English. Return only the corrected sentence.\nInput: {text}"
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return text

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

def predict_static(landmarks):
    rf_proba = rf_model.predict_proba([landmarks])[0]
    idx = np.argmax(rf_proba)
    label = le_static.inverse_transform([idx])[0]
    conf = float(rf_proba[idx])
    # Get top 3 for debug display
    top3_idx = np.argsort(rf_proba)[-3:][::-1]
    top3 = [(le_static.classes_[i], round(rf_proba[i]*100, 1)) for i in top3_idx]
    return label, conf, top3

def predict_dynamic(sequence):
    arr = np.array(sequence).reshape(1, 30, 63)
    proba = lstm_model.predict(arr, verbose=0)[0]
    idx = np.argmax(proba)
    return le_dynamic.inverse_transform([idx])[0], float(proba[idx])

# State
sequence_buffer = []
static_buffer = []
current_letter = ""
current_word = ""
sentence = ""
corrected_sentence = ""
status_msg = ""
status_timer = 0
dynamic_cooldown = 0
DYNAMIC_COOLDOWN = 45
STATIC_THRESHOLD = 0.40  # lowered significantly so letters show
DYNAMIC_THRESHOLD = 0.80
STABILITY_FRAMES = 10
space_count = 0
del_count = 0
is_correcting = False
mode = "LETTER"
last_added_letter = ""
last_added_time = 0
LETTER_COOLDOWN = 2.0
top3_display = []

def run_grammar_correction():
    global corrected_sentence, status_msg, status_timer, is_correcting
    is_correcting = True
    status_msg = "Fixing grammar with Gemini..."
    status_timer = time.time()
    corrected_sentence = correct_grammar(sentence)
    status_msg = "Grammar fixed!"
    status_timer = time.time()
    is_correcting = False

def draw_text_wrapped(img, text, x, y, max_width, font_scale, color, thickness):
    words = text.split(' ')
    line = ''
    line_y = y
    for word in words:
        test = line + word + ' '
        size = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        if size[0] > max_width and line:
            cv2.putText(img, line.strip(), (x, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            line_y += 30
            line = word + ' '
        else:
            line = test
    if line:
        cv2.putText(img, line.strip(), (x, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nSignBridge Running")
print("SPACE - add word | G - grammar | C - clear | BKSP - delete | M - mode | R - restart | Q - quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    canvas = np.zeros((600, 1000, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)

    frame_resized = cv2.resize(frame, (500, 420))
    canvas[80:500, 0:500] = frame_resized

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            temp = canvas[80:500, 0:500].copy()
            mp_draw.draw_landmarks(temp, lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(180, 180, 255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(120, 100, 220), thickness=2))
            canvas[80:500, 0:500] = temp

    hand_detected = False
    if results.multi_hand_landmarks:
        hand_detected = True
        lm = results.multi_hand_landmarks[0]
        coords = []
        for point in lm.landmark:
            coords.extend([point.x, point.y, point.z])
        landmarks = np.array(coords)

        sequence_buffer.append(coords)
        if len(sequence_buffer) > 30:
            sequence_buffer.pop(0)

        if mode == "LETTER":
            s_label, s_conf, top3 = predict_static(landmarks)
            top3_display = top3

            if s_conf >= STATIC_THRESHOLD:
                current_letter = s_label

                if s_label == 'space':
                    space_count += 1
                    if space_count >= 8:
                        if current_word.strip():
                            sentence = (sentence + ' ' + current_word).strip()
                        current_word = ""
                        space_count = 0
                        static_buffer.clear()
                        last_added_letter = ""
                else:
                    space_count = 0

                if s_label == 'del':
                    del_count += 1
                    if del_count >= 8:
                        current_word = current_word[:-1]
                        del_count = 0
                        static_buffer.clear()
                else:
                    del_count = 0

                if s_label not in ['space', 'del']:
                    static_buffer.append(s_label)
                    if len(static_buffer) >= STABILITY_FRAMES:
                        last = static_buffer[-STABILITY_FRAMES:]
                        most_common = max(set(last), key=last.count)
                        count = last.count(most_common)
                        if count >= 8 and most_common.isalpha() and len(most_common) == 1:
                            now = time.time()
                            if most_common != last_added_letter or (now - last_added_time) > LETTER_COOLDOWN:
                                current_word += most_common.lower()
                                current_letter = most_common
                                last_added_letter = most_common
                                last_added_time = now
                                status_msg = f"Added: {most_common.upper()}"
                                status_timer = now
                            static_buffer.clear()
                    if len(static_buffer) > 40:
                        static_buffer.pop(0)
            else:
                current_letter = ""

        elif mode == "WORD":
            current_letter = ""
            top3_display = []
            if len(sequence_buffer) == 30:
                if dynamic_cooldown > 0:
                    dynamic_cooldown -= 1
                else:
                    d_label, d_conf = predict_dynamic(sequence_buffer)
                    if d_conf >= DYNAMIC_THRESHOLD:
                        dynamic_cooldown = DYNAMIC_COOLDOWN
                        sequence_buffer.clear()
                        current_word = d_label
                        status_msg = f"Word: {d_label.upper()}"
                        status_timer = time.time()
    else:
        sequence_buffer.clear()
        static_buffer.clear()
        space_count = 0
        del_count = 0
        current_letter = ""
        top3_display = []
        dynamic_cooldown = max(0, dynamic_cooldown - 1)

    # ── UI ──

    # Title bar
    cv2.rectangle(canvas, (0, 0), (1000, 75), (25, 25, 25), -1)
    cv2.line(canvas, (0, 75), (1000, 75), (60, 60, 60), 1)
    cv2.putText(canvas, "SignBridge", (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (200, 180, 255), 2)
    cv2.putText(canvas, "ASL Translator", (205, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)

    mode_color = (180, 160, 255) if mode == "LETTER" else (100, 220, 150)
    cv2.rectangle(canvas, (650, 18), (790, 55), (35, 35, 35), -1)
    cv2.rectangle(canvas, (650, 18), (790, 55), mode_color, 1)
    cv2.putText(canvas, mode + " MODE", (660, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1)

    dot_color = (100, 220, 150) if hand_detected else (100, 100, 100)
    cv2.circle(canvas, (840, 37), 7, dot_color, -1)
    cv2.putText(canvas, "Hand" if hand_detected else "No Hand", (855, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, dot_color, 1)

    cv2.line(canvas, (505, 80), (505, 595), (50, 50, 50), 1)
    cv2.rectangle(canvas, (510, 80), (995, 595), (22, 22, 22), -1)

    # Big letter display
    cv2.putText(canvas, "LIVE DETECTION", (525, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
    letter_display = current_letter.upper() if current_letter else "-"
    letter_color = (200, 180, 255) if current_letter else (50, 50, 50)
    cv2.putText(canvas, letter_display, (525, 182),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, letter_color, 4)

    # Top 3 predictions debug — very useful
    if top3_display:
        ty = 115
        for i, (lbl, pct) in enumerate(top3_display):
            bar_max = 150
            bar_w = int((pct / 100) * bar_max)
            bx = 720
            bar_color = (200, 180, 255) if i == 0 else (80, 80, 80)
            cv2.rectangle(canvas, (bx, ty - 10), (bx + bar_max, ty + 2), (40, 40, 40), -1)
            cv2.rectangle(canvas, (bx, ty - 10), (bx + bar_w, ty + 2), bar_color, -1)
            cv2.putText(canvas, f"{lbl.upper()} {pct:.0f}%", (bx + bar_max + 5, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, bar_color, 1)
            ty += 22

    cv2.line(canvas, (525, 210), (985, 210), (40, 40, 40), 1)

    # Current word
    cv2.putText(canvas, "CURRENT WORD", (525, 232),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    word_display = current_word + "|" if current_word else "..."
    cv2.putText(canvas, word_display, (525, 262),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 1)

    cv2.line(canvas, (525, 278), (985, 278), (40, 40, 40), 1)

    # Sentence
    cv2.putText(canvas, "SENTENCE", (525, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.rectangle(canvas, (520, 308), (990, 385), (28, 28, 28), -1)
    cv2.rectangle(canvas, (520, 308), (990, 385), (55, 55, 55), 1)
    if sentence:
        draw_text_wrapped(canvas, sentence, 530, 335, 445, 0.58, (240, 240, 240), 1)
    else:
        cv2.putText(canvas, "Sign to start...", (530, 345),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

    cv2.line(canvas, (525, 393), (985, 393), (40, 40, 40), 1)

    # Gemini
    cv2.putText(canvas, "GEMINI CORRECTED", (525, 415),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 130), 1)
    cv2.rectangle(canvas, (520, 422), (990, 492), (24, 32, 26), -1)
    cv2.rectangle(canvas, (520, 422), (990, 492), (50, 90, 60), 1)
    if corrected_sentence:
        draw_text_wrapped(canvas, corrected_sentence, 530, 450, 445, 0.58, (160, 240, 180), 1)
    elif is_correcting:
        cv2.putText(canvas, "Calling Gemini...", (530, 455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    else:
        cv2.putText(canvas, "Press G to correct grammar", (530, 455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 55, 55), 1)

    cv2.line(canvas, (525, 500), (985, 500), (40, 40, 40), 1)

    controls = [
        ("SPC", "push word to sentence"),
        ("M", "Letter / Word mode"),
        ("G", "Gemini grammar fix"),
        ("BKSP", "delete letter"),
        ("R", "restart  |  C clear  |  Q quit"),
    ]
    cy = 515
    for key, desc in controls:
        cv2.rectangle(canvas, (525, cy - 2), (565, cy + 14), (38, 38, 38), -1)
        cv2.putText(canvas, key, (527, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 180, 255), 1)
        cv2.putText(canvas, desc, (572, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1)
        cy += 18

    # Status
    cv2.rectangle(canvas, (0, 505), (500, 535), (25, 25, 25), -1)
    if status_msg and (time.time() - status_timer) < 3:
        cv2.putText(canvas, status_msg, (10, 525),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 150), 1)
    else:
        cv2.putText(canvas, "RF Model + LSTM  |  Hold sign 1 sec", (10, 525),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1)

    cv2.rectangle(canvas, (0, 535), (500, 600), (20, 20, 20), -1)
    cv2.putText(canvas, "Hold sign still to register  |  Good light = better accuracy", (10, 558),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1)
    cv2.putText(canvas, "SPACE = word done  |  M = switch mode  |  G = fix grammar", (10, 580),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (55, 55, 55), 1)

    cv2.imshow("SignBridge — ASL Translator", canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        if current_word.strip():
            sentence = (sentence + ' ' + current_word).strip()
            current_word = ""
            static_buffer.clear()
            last_added_letter = ""
            status_msg = "Word added to sentence!"
            status_timer = time.time()
    elif key == 8:
        if current_word:
            current_word = current_word[:-1]
            status_msg = "Letter deleted"
            status_timer = time.time()
    elif key == ord('g'):
        if sentence and not is_correcting:
            t = threading.Thread(target=run_grammar_correction, daemon=True)
            t.start()
    elif key == ord('c'):
        current_word = ""
        sentence = ""
        corrected_sentence = ""
        current_letter = ""
        static_buffer.clear()
        sequence_buffer.clear()
        last_added_letter = ""
        status_msg = "Cleared!"
        status_timer = time.time()
    elif key == ord('m'):
        mode = "WORD" if mode == "LETTER" else "LETTER"
        current_letter = ""
        current_word = ""
        static_buffer.clear()
        sequence_buffer.clear()
        dynamic_cooldown = 0
        last_added_letter = ""
        top3_display = []
        status_msg = f"Switched to {mode} mode"
        status_timer = time.time()
    elif key == ord('r'):
        current_word = ""
        sentence = ""
        corrected_sentence = ""
        current_letter = ""
        static_buffer.clear()
        sequence_buffer.clear()
        dynamic_cooldown = 0
        space_count = 0
        del_count = 0
        last_added_letter = ""
        top3_display = []
        mode = "LETTER"
        status_msg = "Restarted!"
        status_timer = time.time()

cap.release()
cv2.destroyAllWindows()