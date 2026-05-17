import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

SIGNS = ['hello', 'thankyou', 'sorry', 'please',
         'help', 'yes', 'no', 'stop', 'more', 'water',
         'good', 'bad', 'want', 'come', 'go',
         'i', 'you', 'what', 'where', 'name']

SEQUENCES = 30
FRAMES = 30
SAVE_PATH = "data_collection/dynamic_signs"

os.makedirs(SAVE_PATH, exist_ok=True)
for sign in SIGNS:
    os.makedirs(os.path.join(SAVE_PATH, sign), exist_ok=True)

cap = cv2.VideoCapture(0)

def extract_landmarks(results):
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        coords = []
        for point in lm.landmark:
            coords.extend([point.x, point.y, point.z])
        return coords
    return [0.0] * 63

for sign in SIGNS:

    # ── PAUSE SCREEN before every sign ──
    print(f"\nReady for: {sign.upper()} — press SPACE in webcam window to start")
    waiting = True
    while waiting:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, f"NEXT: {sign.upper()}", (w//2-160, h//2-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)
        cv2.putText(frame, "Press SPACE when ready", (w//2-200, h//2+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("SignBridge", frame)
        key = cv2.waitKey(30)
        if key == 32:  # SPACE
            waiting = False

    # ── Record 30 sequences ──
    for seq in range(SEQUENCES):

        # Countdown
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame,
                                       results.multi_hand_landmarks[0],
                                       mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"{sign.upper()}  {seq+1}/{SEQUENCES}", (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, str(countdown), (w//2-20, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 165, 255), 5)
            cv2.imshow("SignBridge", frame)
            cv2.waitKey(1000)

        # Record frames
        sequence = []
        for frame_num in range(FRAMES):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame,
                                       results.multi_hand_landmarks[0],
                                       mp_hands.HAND_CONNECTIONS)

            landmarks = extract_landmarks(results)
            sequence.append(landmarks)

            # Progress bar
            progress = int((frame_num / FRAMES) * 400)
            cv2.rectangle(frame, (15, 15), (415, 38), (50, 50, 50), -1)
            cv2.rectangle(frame, (15, 15), (15 + progress, 38), (0, 0, 255), -1)
            cv2.putText(frame, f"{sign.upper()}  {seq+1}/{SEQUENCES}", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "RECORDING", (w-200, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("SignBridge", frame)
            cv2.waitKey(33)

        # Save
        np.save(os.path.join(SAVE_PATH, sign, f"seq_{seq}.npy"), np.array(sequence))
        time.sleep(0.3)

    print(f"Done: {sign.upper()}")

cap.release()
cv2.destroyAllWindows()
print("\nAll signs recorded!")