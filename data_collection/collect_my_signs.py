import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

LABELS = ['A','B','C','D','E','F','G','H','I','K','L','M',
          'N','O','P','Q','R','S','T','U','V','W','X','Y',]
SAMPLES_PER_CLASS = 200
OUTPUT_CSV = "data_collection/my_signs.csv"

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    header = [f"x{i}" for i in range(21)] + \
             [f"y{i}" for i in range(21)] + \
             [f"z{i}" for i in range(21)] + ["label"]
    writer.writerow(header)

cap = cv2.VideoCapture(0)

for label in LABELS:
    print(f"\nNext: {label} — get ready and press SPACE to start recording")

    # Wait for spacebar
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"NEXT SIGN: {label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, "Press SPACE when ready", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Collect My Signs", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    count = 0
    print(f"Recording {label}... hold the sign steady")

    while count < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            coords = []
            for point in lm.landmark:
                coords.extend([point.x, point.y, point.z])

            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(coords + [label])

            count += 1

        # Progress bar
        progress = int((count / SAMPLES_PER_CLASS) * 400)
        cv2.rectangle(frame, (10, 10), (410, 35), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 10), (10 + progress, 35), (0, 255, 0), -1)
        cv2.putText(frame, f"{label}: {count}/{SAMPLES_PER_CLASS}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collect My Signs", frame)
        cv2.waitKey(1)

    print(f"Done: {label}")

cap.release()
cv2.destroyAllWindows()
print(f"\nAll done. Saved to {OUTPUT_CSV}")