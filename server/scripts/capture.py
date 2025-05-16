from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import requests
import uuid

# Initialize Flask app
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MediaPipe Holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

ROWS_PER_FRAME = 543  # Define the number of rows per frame for processing

def create_frame_landmarks_df(results, frame, xyz):
    xyz_skel = xyz[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()

    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()

    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    face = face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face')
    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')
    left_hand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')
    right_hand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
    landmarks = xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)
    return landmarks

def process_video(video_path, xyz):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    all_landmarks = []
    frame_num = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_num += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            landmarks = create_frame_landmarks_df(results, frame_num, xyz)
            all_landmarks.append(landmarks)

    cap.release()
    return all_landmarks

def create_frame_windows(landmarks_df, window_size=20, overlap=3):
    frames = landmarks_df['frame'].unique()
    frames.sort()

    windows = []
    stride = window_size - overlap

    for start_idx in range(0, len(frames) - window_size + 1, stride):
        window_frames = frames[start_idx:start_idx + window_size]
        window_data = landmarks_df[landmarks_df['frame'].isin(window_frames)]

        n_frames = len(window_frames)
        data_columns = ['x', 'y', 'z']
        window_array = []

        for frame in window_frames:
            frame_data = window_data[window_data['frame'] == frame][data_columns].values
            window_array.append(frame_data)

        if len(window_array) == window_size:
            window_array = np.array(window_array).reshape(window_size, ROWS_PER_FRAME, len(data_columns))
            windows.append(window_array.astype(np.float32))

    return windows

@app.route('/api/process-video', methods=['POST'])
def process_video_api():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded video file
    video_filename = f"{uuid.uuid4().hex}.webm"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    video_file.save(video_path)

    try:
        # Load the XYZ skeleton data
        xyz = pd.read_parquet("./1460359.parquet")

        # Process the video
        landmarks = process_video(video_path, xyz)
        if isinstance(landmarks, dict) and 'error' in landmarks:
            return jsonify(landmarks), 500

        if landmarks:
            combined_landmarks = pd.concat(landmarks).reset_index(drop=True)
            combined_landmarks.to_parquet('output.parquet')

            # Load TensorFlow Lite model
            interpreter = tf.lite.Interpreter("model.tflite")
            interpreter.allocate_tensors()
            prediction_fn = interpreter.get_signature_runner("serving_default")

            train = pd.read_csv("./train.csv")
            train['sign_ord'] = train['sign'].astype('category').cat.codes

            SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
            ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

            windows = create_frame_windows(combined_landmarks, window_size=5, overlap=3)
            window_predictions = []
            for i, window_data in enumerate(windows):
                prediction = prediction_fn(inputs=window_data)
                sign_idx = prediction['outputs'].argmax()
                if sign_idx not in ORD2SIGN:
                    continue

                sign_name = ORD2SIGN[sign_idx]
                confidence = prediction['outputs'].max()
                if confidence >= 0.35:
                    window_predictions.append((i, sign_name, confidence))

            if window_predictions:
                sign_counts = {}
                for _, sign, _ in window_predictions:
                    sign_counts[sign] = sign_counts.get(sign, 0) + 1

                most_common_sign = max(sign_counts.items(), key=lambda x: x[1])[0]

                # --- Ollama Sentence Generation ---
                final_signs = [sign for _, sign, _ in window_predictions]
                if final_signs:
                    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
                    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

                    sign_string = ", ".join(final_signs)
                    prompt = f"Translate the following sequence of sign language words into a simple English phrase. Signs: {sign_string}"

                    ollama_payload = {
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False
                    }

                    try:
                        response = requests.post(OLLAMA_URL, json=ollama_payload, timeout=60)
                        response.raise_for_status()
                        ollama_data = response.json()
                        generated_sentence = ollama_data.get('response', '').strip()
                        return jsonify({'recognizedText': most_common_sign, 'generatedSentence': generated_sentence})
                    except Exception as e:
                        return jsonify({'error': 'Error during Ollama request', 'details': str(e)}), 500
                # --- End Ollama Sentence Generation ---

                return jsonify({'recognizedText': most_common_sign})
            return jsonify({'recognizedText': 'No predictions made.'})
        else:
            return jsonify({'error': 'No landmarks captured or capture failed.'}), 500
    finally:
        # Clean up the uploaded video file
        if os.path.exists(video_path):
            print("Hi i am here")
            # os.remove(video_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)