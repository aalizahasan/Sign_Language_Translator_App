import os
import cv2
import numpy as np
import mediapipe as mp

DATASET_PATH = "datasets/my_videos"
OUTPUT_PATH = "datasets/features"

os.makedirs(OUTPUT_PATH, exist_ok=True)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, face, lh, rh])  
    # total length = 1662

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        keypoints = extract_keypoints(results)
        all_keypoints.append(keypoints)

    cap.release()
    return np.array(all_keypoints)


for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing class: {label}")

    for video_file in os.listdir(label_path):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(label_path, video_file)
        features = extract_landmarks(video_path)

        if features.size == 0:
            print(f"❌ No landmarks for {video_file}")
            continue

        save_name = f"{label}_{video_file.replace('.mp4','')}.npy"
        save_path = os.path.join(OUTPUT_PATH, save_name)

        np.save(save_path, features)
        print(f"✅ Saved: {save_path}")
