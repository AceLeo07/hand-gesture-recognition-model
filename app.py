import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tempfile import NamedTemporaryFile

model = load_model('./hand_gesture_recognition_model.h5')

gesture_names = {
    0: 'Index Pointing Up',
    1: 'Palm Down',
    2: 'Fist',
    3: 'Thumbs Down',
    4: 'Thumbs Up',
    5: 'Palm Up',
    6: 'Victory',
    7: 'Stop',
    8: 'OK',
    9: 'Call Me'
}

def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # Resize frame to match model input
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=-1)  # Add channel dimension (64, 64, 1)
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension (1, 64, 64, 1)
    return frame

st.title("Hand Gesture Recognition from Uploaded Video")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = preprocess_frame(frame)

        prediction = model.predict(processed_frame)
        predicted_class = np.argmax(prediction, axis=1)[0]
        gesture = gesture_names[predicted_class]  # Map class index to gesture name

        cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        stframe.image(frame, channels="BGR")
    
    cap.release()
else:
    st.write("Please upload a video file.")
