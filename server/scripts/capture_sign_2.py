import os

os.environ["GLOG_minloglevel"] = "3"  # Suppress INFO and WARNING logs from MediaPipe
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs if used

import cv2
import mediapipe as mp
import pandas as pd
import tensorflow as tf
import numpy as np
import time
import requests
import sys
import json
import os
import subprocess

# import google.generativeai as genai

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Define constant early to avoid errors
ROWS_PER_FRAME = 543


def create_frame_landmarks_df(results, frame, xyz):
    xyz_skel = (
        xyz[["type", "landmark_index"]].drop_duplicates().reset_index(drop=True).copy()
    )

    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()

    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]

    face = (
        face.reset_index()
        .rename(columns={"index": "landmark_index"})
        .assign(type="face")
    )
    pose = (
        pose.reset_index()
        .rename(columns={"index": "landmark_index"})
        .assign(type="pose")
    )
    left_hand = (
        left_hand.reset_index()
        .rename(columns={"index": "landmark_index"})
        .assign(type="left_hand")
    )
    right_hand = (
        right_hand.reset_index()
        .rename(columns={"index": "landmark_index"})
        .assign(type="right_hand")
    )

    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)

    landmarks = xyz_skel.merge(landmarks, on=["type", "landmark_index"], how="left")
    landmarks = landmarks.assign(frame=frame)
    return landmarks


def process_video(video_path, xyz):
    """Process a video file and extract landmarks."""
    all_landmarks = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(json.dumps({"error": "Could not open video file."}))
            return all_landmarks

        with mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            frame_num = 0
            while cap.isOpened():
                frame_num += 1
                success, image = cap.read()
                if not success:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                landmarks = create_frame_landmarks_df(results, frame_num, xyz)
                all_landmarks.append(landmarks)

        cap.release()
        return all_landmarks

    except Exception as e:
        print(json.dumps({"error": f"Exception during video processing: {str(e)}"}))
        return all_landmarks


def create_frame_windows(landmarks_df, window_size=20, overlap=3):
    """
    Split landmarks data into overlapping windows.
    """
    frames = landmarks_df["frame"].unique()
    frames.sort()

    windows = []
    stride = window_size - overlap

    for start_idx in range(0, len(frames) - window_size + 1, stride):
        window_frames = frames[start_idx : start_idx + window_size]
        window_data = landmarks_df[landmarks_df["frame"].isin(window_frames)]

        n_frames = len(window_frames)
        data_columns = ["x", "y", "z"]
        window_array = []

        for frame in window_frames:
            frame_data = window_data[window_data["frame"] == frame][data_columns].values
            window_array.append(frame_data)

        if len(window_array) == window_size:
            window_array = np.array(window_array).reshape(
                window_size, ROWS_PER_FRAME, len(data_columns)
            )
            windows.append(window_array.astype(np.float32))

    return windows


if __name__ == "__main__":
    try:
        # Check if a video path was provided as a command line argument
        if len(sys.argv) < 2:
            print(json.dumps({"error": "No video path provided"}))
            sys.exit(1)

        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(json.dumps({"error": f"Video file not found: {video_path}"}))
            sys.exit(1)

        # Get the directory of this script to resolve relative paths
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load reference data - use absolute path
        xyz_path = os.path.join(script_dir, "1460359.parquet")
        if not os.path.exists(xyz_path):
            print(json.dumps({"error": f"Reference data not found: {xyz_path}"}))
            sys.exit(1)

        xyz = pd.read_parquet(xyz_path)

        # Process the video file
        landmarks = process_video(video_path, xyz)

        if not landmarks:
            print(json.dumps({"recognizedText": "No landmarks detected in video"}))
            sys.exit(0)

        # Combine all landmarks into a DataFrame
        combined_landmarks = pd.concat(landmarks).reset_index(drop=True)

        # Load TensorFlow Lite model - use absolute path
        model_path = os.path.join(script_dir, "model.tflite")
        if not os.path.exists(model_path):
            print(json.dumps({"error": f"Model file not found: {model_path}"}))
            sys.exit(1)

        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        prediction_fn = interpreter.get_signature_runner("serving_default")

        # Load training data - use absolute path
        train_path = os.path.join(script_dir, "train.csv")
        if not os.path.exists(train_path):
            print(json.dumps({"error": f"Training data not found: {train_path}"}))
            sys.exit(1)

        train = pd.read_csv(train_path)
        train["sign_ord"] = train["sign"].astype("category").cat.codes

        SIGN2ORD = train[["sign", "sign_ord"]].set_index("sign").squeeze().to_dict()
        ORD2SIGN = train[["sign_ord", "sign"]].set_index("sign_ord").squeeze().to_dict()

        # Create windows of frames
        windows = create_frame_windows(combined_landmarks, window_size=5, overlap=3)

        # Process each window
        window_predictions = []
        for i, window_data in enumerate(windows):
            # Get prediction for window
            prediction = prediction_fn(inputs=window_data)
            sign_idx = prediction["outputs"].argmax()
            # Check if sign_idx is valid before accessing ORD2SIGN
            if sign_idx not in ORD2SIGN:
                continue

            sign_name = ORD2SIGN[sign_idx]
            confidence = prediction["outputs"].max()

            # Condition 1 & 2: Check confidence threshold and NaN
            if confidence < 0.35 or np.isnan(confidence):
                continue

            # Condition 3: Check for consecutive duplicates
            if window_predictions and window_predictions[-1][1] == sign_name:
                continue

            # If all checks pass, append the prediction
            window_predictions.append((i, sign_name, confidence))

        # Find most common prediction
        if window_predictions:
            sign_counts = {}
            for _, sign, _ in window_predictions:
                sign_counts[sign] = sign_counts.get(sign, 0) + 1

            most_common_sign = max(sign_counts.items(), key=lambda x: x[1])[0]

            # --- Ollama Sentence Generation ---
            # final_signs = [sign for _, sign, _ in window_predictions]
            final_signs = [sign for _, sign, _ in window_predictions]

            if final_signs:
                sign_string = ", ".join(final_signs)

                try:
                    # Call Gemini subprocess
                    result = subprocess.run(
                        [
                            "gemini-env\\Scripts\\python.exe",
                            ".\server\scripts\generate_with_gemini.py",
                            sign_string,
                        ],
                        capture_output=True,
                        timeout=60,
                    )
                    # print("gemini")

                    response = result.stdout.decode().strip()
                    stderr_output = result.stderr.decode().strip()
                    debug_info = {
                        "gemini_stdout": response,
                        "gemini_stderr": stderr_output,
                        "gemini_exit_code": result.returncode
                    }
                    
                    gemini_output = json.loads(response)

                    generated_sentence = gemini_output.get("sentence", "")

                    # Return results as JSON
                    result = {
                        "recognizedText": most_common_sign,
                        "generatedSentence": generated_sentence,
                        "allSigns": final_signs,
                        "debug": debug_info
                    }
                    print(json.dumps(result))

                except Exception as e:
                    # Return just the recognized sign if Gemini call fails
                    print(
                        json.dumps(
                            {
                                "recognizedText": most_common_sign,
                                "allSigns": final_signs,
                                "error": "Failed to parse Gemini response",
                                "debug": debug_info
                            }
                        )
                    )
            else:
                print(json.dumps({"recognizedText": generated_sentence}))
        else:
            print(json.dumps({"recognizedText": "No signs recognized"}))

        #     if final_signs:
        #         OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        #         OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

        #         sign_string = ", ".join(final_signs)
        #         # New prompt focused on simplicity and sign language context
        #         prompt = f"Translate the following sequence of sign language words into a simple English phrase. Use only the given words and the absolute minimum necessary connecting words (like 'on', 'in', 'at'). Signs: {sign_string}"

        #         try:
        #             response = requests.post(OLLAMA_URL, json={
        #                 "model": OLLAMA_MODEL,
        #                 "prompt": prompt,
        #                 "stream": False
        #             }, timeout=60)

        #             response.raise_for_status()
        #             ollama_data = response.json()
        #             generated_sentence = ollama_data.get('response', '').strip()

        #             # Return results as JSON
        #             result = {
        #                 "recognizedText": most_common_sign,
        #                 "generatedSentence": generated_sentence,
        #                 "allSigns": final_signs
        #             }
        #             print(json.dumps(result))

        #         except Exception as e:
        #             # Return just the recognized sign if Ollama fails
        #             print(json.dumps({
        #                 "recognizedText": most_common_sign,
        #                 "allSigns": final_signs
        #             }))
        #     else:
        #         print(json.dumps({"recognizedText": most_common_sign}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
