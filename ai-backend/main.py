from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tempfile
import os
import json
import base64
import mediapipe as mp
import imageio # <--- NEW LIBRARY
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "sign_model.keras"
model = load_model(MODEL_PATH)

LABELS_PATH = "labels.json"
label_map = {}
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        data = json.load(f)
        label_map = {v: k for k, v in data.items()}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
holistic = mp_holistic.Holistic(static_image_mode=False)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        cap = cv2.VideoCapture(temp_path)
        frames_features = []
        frames_with_hands = 0
        
        # List to store frames for the output GIF
        replay_frames = [] 

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip and Convert
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # --- DRAWING LOGIC FOR REPLAY ---
            # We draw on EVERY frame now, not just one
            debug_frame = frame.copy()
            if results.left_hand_landmarks or results.right_hand_landmarks:
                frames_with_hands += 1
                
                # Draw landmarks
                mp_drawing.draw_landmarks(debug_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(debug_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(debug_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Convert BGR (OpenCV) to RGB (ImageIO) for the GIF
            rgb_debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
            
            # Resize to make the GIF smaller/faster (optional but recommended)
            small_frame = cv2.resize(rgb_debug_frame, (320, 240))
            replay_frames.append(small_frame)
            # -------------------------------

            keypoints = extract_keypoints(results)
            frames_features.append(keypoints)

        cap.release()
        os.unlink(temp_path)

        if frames_with_hands == 0:
            return {"prediction": "No Hands Detected", "confidence": 0.0, "replay_gif": None}

        # --- GENERATE GIF ---
        # Create a temporary GIF in memory
        gif_path = tempfile.mktemp(suffix=".gif")
        # Write frames to GIF (fps=10 is good for preview)
        imageio.mimsave(gif_path, replay_frames, fps=10)
        
        # Convert GIF to Base64
        with open(gif_path, "rb") as gif_file:
            gif_base64 = base64.b64encode(gif_file.read()).decode('utf-8')
        
        os.remove(gif_path) # Clean up
        # --------------------

        input_data = np.mean(frames_features, axis=0)
        input_data = np.expand_dims(input_data, axis=0)
        prediction = model.predict(input_data)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        sign_result = label_map.get(class_index, "Unknown")

        return {
            "prediction": sign_result, 
            "confidence": confidence,
            "replay_gif": gif_base64 # Sending the moving video back!
        }

    except Exception as e:
        return {"error": str(e)}