import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("Loading data...")
df = pd.read_csv("data_collection/asl_landmarks.csv")
df = df[df['label'] != 'nothing']

print(f"Total samples: {len(df)}")
print(f"Classes: {df['label'].nunique()}")
print(df['label'].unique())

X = df.drop('label', axis=1).values
y = df['label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Static classes ({len(le.classes_)}): {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Random Forest ──
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")

# ── Neural Network ──
print("\nTraining Neural Network...")
num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

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
    keras.layers.Dense(num_classes, activation='softmax')
])

nn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

nn_model.fit(
    X_train_scaled, y_train_cat,
    epochs=100,
    batch_size=64,
    validation_data=(X_test_scaled, y_test_cat),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

nn_loss, nn_acc = nn_model.evaluate(X_test_scaled, y_test_cat, verbose=0)
print(f"Neural Network Accuracy: {nn_acc*100:.2f}%")

# ── LSTM ──
print("\nTraining LSTM...")
SIGNS = sorted(os.listdir('data_collection/dynamic_signs'))
X_dyn = []
y_dyn = []

for sign in SIGNS:
    sign_path = f'data_collection/dynamic_signs/{sign}'
    for seq_file in os.listdir(sign_path):
        sequence = np.load(f'{sign_path}/{seq_file}')
        X_dyn.append(sequence)
        y_dyn.append(sign)

X_dyn = np.array(X_dyn)
y_dyn = np.array(y_dyn)

le_dynamic = LabelEncoder()
y_dyn_encoded = le_dynamic.fit_transform(y_dyn)
print(f"Dynamic classes ({len(le_dynamic.classes_)}): {le_dynamic.classes_}")

y_dyn_cat = to_categorical(y_dyn_encoded)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_dyn, y_dyn_cat, test_size=0.2, random_state=42, stratify=y_dyn_encoded
)

lstm_model = keras.Sequential([
    keras.layers.Input(shape=(30, 63)),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dropout(0.3),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.Dropout(0.3),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(le_dynamic.classes_), activation='softmax')
])

lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop2 = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001)

lstm_model.fit(
    X_tr, y_tr,
    epochs=150,
    batch_size=32,
    validation_data=(X_te, y_te),
    callbacks=[early_stop2, reduce_lr2],
    verbose=1
)

lstm_loss, lstm_acc = lstm_model.evaluate(X_te, y_te, verbose=0)
print(f"LSTM Accuracy: {lstm_acc*100:.2f}%")

# ── Save everything ──
print("\nSaving all models...")

with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/le_dynamic.pkl', 'wb') as f:
    pickle.dump(le_dynamic, f)

nn_model.save_weights('models/nn_weights.weights.h5')
lstm_model.save_weights('models/lstm_weights.weights.h5')

print("\nAll models saved successfully")
print(f"Static classes: {len(le.classes_)}")
print(f"Dynamic classes: {len(le_dynamic.classes_)}")
print(f"Random Forest: {rf_acc*100:.2f}%")
print(f"Neural Network: {nn_acc*100:.2f}%")
print(f"LSTM: {lstm_acc*100:.2f}%")