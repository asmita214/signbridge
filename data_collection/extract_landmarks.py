import os
import csv
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

DATASET_PATH = "asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV = "data_collection/asl_landmarks.csv"

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    header = [f"x{i}" for i in range(21)] + \
             [f"y{i}" for i in range(21)] + \
             [f"z{i}" for i in range(21)] + ["label"]
    writer.writerow(header)

total = 0
skipped = 0

for label in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_path):
        continue

    print(f"Processing: {label}")

    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        image = cv2.imread(img_path)

        if image is None:
            skipped += 1
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            x_vals = [lm.x for lm in landmarks.landmark]
            y_vals = [lm.y for lm in landmarks.landmark]
            z_vals = [lm.z for lm in landmarks.landmark]
            row = x_vals + y_vals + z_vals + [label]

            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            total += 1
        else:
            skipped += 1

print(f"Done. Total saved: {total}, Skipped: {skipped}")
print(f"CSV saved at: {OUTPUT_CSV}")