import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -------------------------
# 1. Setup Paths
# -------------------------
FEATURES_PATH = "datasets/features"
MODEL_PATH = "sign_model.keras"
LABELS_PATH = "labels.json"

# -------------------------
# 2. Load Features
# -------------------------
X = []
y = []
labels = {}
label_index = 0

print("Loading data...")

for file in os.listdir(FEATURES_PATH):
    if file.endswith(".npy"):
        # Extract label from filename (e.g., "sorry_video1.npy" -> "sorry")
        label = file.split("_")[0]

        if label not in labels:
            labels[label] = label_index
            label_index += 1

        data = np.load(os.path.join(FEATURES_PATH, file))

        if data.size == 0:
            continue

        # Average over frames to get shape (1662,)
        # Note: If your feature extractor changed, this shape might be slightly different.
        # We auto-detect shape below.
        X.append(np.mean(data, axis=0))
        y.append(labels[label])

X = np.array(X)
y_original = np.array(y) # Keep original integers for charts
y = to_categorical(y_original) # One-hot encode for training

print(f"Feature shape: {X.shape}")
print(f"Classes: {labels}")
print(f"Total samples: {len(X)}")

if len(X) < 10:
    print("⚠️ CRITICAL WARNING: You have very little data! Record more videos.")

# -------------------------
# 3. REPORT: Data Distribution Chart
# -------------------------
# This generates the first graph for your report
unique, counts = np.unique(y_original, return_counts=True)
plt.figure(figsize=(8, 5))
plt.bar(list(labels.keys()), counts, color=['blue', 'orange', 'green'])
plt.title("Dataset Distribution")
plt.xlabel("Signs")
plt.ylabel("Number of Videos")
plt.savefig("dataset_chart.png")
print("✅ Saved 'dataset_chart.png' for your report.")

# -------------------------
# 4. Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 5. Build Model
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
# 6. Train Model
# -------------------------
print("\nStarting Training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=4
)

# -------------------------
# 7. REPORT: Evaluation Metrics
# -------------------------
print("\nGenerating Evaluation Report...")

# Predict on Test Data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=list(labels.keys())))

# Generate Confusion Matrix Image
try:
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(labels.keys()), 
                yticklabels=list(labels.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("✅ Saved 'confusion_matrix.png' for your report.")
except Exception as e:
    print(f"Could not save confusion matrix (Needs more test data): {e}")

# -------------------------
# 8. Save Model
# -------------------------
model.save(MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")

with open(LABELS_PATH, "w") as f:
    json.dump(labels, f)
print(f"✅ Labels saved as {LABELS_PATH}")