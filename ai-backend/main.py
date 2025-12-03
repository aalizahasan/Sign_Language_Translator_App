# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tempfile
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

app = FastAPI(title="Sign Language Video Prediction")

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Load trained model =====
MODEL_PATH = "sign_model.keras"
model = load_model(MODEL_PATH)

# ===== Load class labels from dataset folder =====
DATASET_PATH = "datasets/my_videos"
labels = sorted(os.listdir(DATASET_PATH))

# ===== Mediapipe Holistic setup =====
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# ===== Keypoints extraction function =====
def extract_keypoints(results):
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, face, lh, rh])



# ===== Video prediction endpoint =====
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    try:
        # ===== Safe temporary file handling =====
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        cap = cv2.VideoCapture(temp_path)
        features = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            features.append(keypoints)

        cap.release()
        os.unlink(temp_path)  # delete temp file

        # ===== No landmarks found =====
        if len(features) == 0:
            return {"prediction": "❌ No landmarks found"}

        # ===== Prepare features for model =====
        # If model is fully connected: collapse sequence by mean
        features = np.expand_dims(np.mean(features, axis=0), axis=0)
        # If model expects sequences (LSTM), replace with:
        # features = np.expand_dims(np.array(features), axis=0)

        prediction = model.predict(features)
        sign = labels[np.argmax(prediction)]

        return {"prediction": sign}

    except Exception as e:
        return {"error": str(e)}
