import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load Kaggle data
print("Loading Kaggle data...")
df_kaggle = pd.read_csv("data_collection/asl_landmarks.csv")
df_kaggle = df_kaggle[df_kaggle["label"] != "nothing"]
df_kaggle = df_kaggle[df_kaggle["label"] != "space"]
df_kaggle = df_kaggle[df_kaggle["label"] != "del"]
print(f"Kaggle samples: {len(df_kaggle)}")

# Load your own collected data
print("Loading your own data...")
df_mine = pd.read_csv("data_collection/my_signs.csv")
print(f"Your samples: {len(df_mine)}")

# Mirror function for left hand support
def mirror_landmarks(rows):
    mirrored = rows.copy()
    for i in range(21):
        mirrored[:, i] = 1.0 - rows[:, i]
    return mirrored

X_mine = df_mine.drop("label", axis=1).values
y_mine = df_mine["label"].values

# Create mirrored left hand version
X_mine_mirrored = mirror_landmarks(X_mine.copy())
y_mine_mirrored = y_mine.copy()

# Kaggle data
X_kaggle = df_kaggle.drop("label", axis=1).values
y_kaggle = df_kaggle["label"].values

# Also mirror Kaggle data
X_kaggle_mirrored = mirror_landmarks(X_kaggle.copy())
y_kaggle_mirrored = y_kaggle.copy()

# Combine everything
X_all = np.vstack([X_kaggle, X_kaggle_mirrored, X_mine, X_mine_mirrored])
y_all = np.concatenate([y_kaggle, y_kaggle_mirrored, y_mine, y_mine_mirrored])

print(f"Total combined samples: {len(X_all)}")

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y_all)
print(f"Classes ({len(le.classes_)}): {le.classes_}")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# Train RF
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_features="log2",
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

acc = accuracy_score(y_test, rf.predict(X_test))
print(f"\nAccuracy: {acc*100:.2f}%")
print("\nPer class report:")
print(classification_report(y_test, rf.predict(X_test), target_names=le.classes_))

# Save
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nModels saved to models/ folder")
print("Now run python app.py to test")