from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import tempfile
import os

app = FastAPI()

# ✅ CORS allow for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ya frontend IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model
MODEL_PATH = "sign_model.keras"
model = load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Dataset labels
DATASET_PATH = "datasets/my_videos"
labels = sorted(os.listdir(DATASET_PATH))

# Extract landmarks from video file
def extract_landmarks_from_file(file_path):
    cap = cv2.VideoCapture(file_path)
    all_landmarks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                all_landmarks.append(landmarks)

    cap.release()
    return np.array(all_landmarks)

@app.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    try:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        features = extract_landmarks_from_file(tmp_path)

        # Delete temp file after reading
        os.remove(tmp_path)

        if features.shape[0] == 0:
            return {"error": "No hand detected in video"}

        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)

        return {"predicted_sign": labels[predicted_class]}

    except Exception as e:
        return {"error": str(e)}
