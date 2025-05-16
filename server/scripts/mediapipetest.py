import mediapipe as mp

mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic() as holistic:
    print("Mediapipe Holistic initialized successfully!")