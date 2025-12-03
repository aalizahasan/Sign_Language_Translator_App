import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -------------------------
# Paths
# -------------------------
FEATURES_PATH = "datasets/features"
MODEL_PATH = "sign_model.keras"
LABELS_PATH = "labels.json"

# -------------------------
# Load Features
# -------------------------
X = []
y = []
labels = {}
label_index = 0

for file in os.listdir(FEATURES_PATH):
    if file.endswith(".npy"):
        label = file.split("_")[0]

        if label not in labels:
            labels[label] = label_index
            label_index += 1

        data = np.load(os.path.join(FEATURES_PATH, file))

        if data.size == 0:
            continue

        # Average over temporal dimension if present
        X.append(np.mean(data, axis=0))
        y.append(labels[label])

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)

EXPECTED_FEATURES = 1629  # holistic: pose+face+lh+rh

if X.shape[1] != EXPECTED_FEATURES:
    print("⚠️ WARNING: Feature size mismatch!")
    print("Expected:", EXPECTED_FEATURES)
    print("Found:", X.shape[1])


print("Classes:", labels)
print("Total samples:", len(X))

# One-hot encoding
y = to_categorical(y)

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Model
# -------------------------
model = Sequential([
    Dense(512, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.4),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dense(y.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Train
# -------------------------
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=4
)

# -------------------------
# Save Model & Labels
# -------------------------
model.save(MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")

with open(LABELS_PATH, "w") as f:
    json.dump(labels, f)
print(f"✅ Labels saved as {LABELS_PATH}")
