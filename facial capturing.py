import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import datetime

# Setup MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

st.title("📷 Real-Time 3D Face Detection & Data Capture")

# Sidebar camera selection
camera_index = st.sidebar.selectbox("Choose camera index", [0, 1, 2], index=0)

# 📝 Custom file name input
custom_name = st.sidebar.text_input("Enter CSV file name (no extension)", "face_data_session")

# 📸 Start and Stop buttons
run_detection = st.sidebar.button("Start Detection")
stop_detection = st.sidebar.button("Stop Detection")

# Frame display window
FRAME_WINDOW = st.image([])

# Session state setup
if "run" not in st.session_state:
    st.session_state.run = False

# Handling detection start and stop
if run_detection:
    st.session_state.run = True
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Combine custom name with timestamp
    base_name = custom_name.strip() if custom_name.strip() else "face_data_session"
    st.session_state.data_file = f"{base_name}_{timestamp}.csv"

if stop_detection:
    st.session_state.run = False

# 📁 Current filename
data_file = st.session_state.get("data_file", "face_data_default.csv")

# Run detection if the button is pressed
if st.session_state.run:
    st.success(f"Face detection started... Saving to `{data_file}`")

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        st.error("❌ Could not open the selected camera.")
        st.session_state.run = False
    else:
        # Create CSV and header if it doesn't exist
        if not os.path.exists(data_file):
            with open(data_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = ['timestamp'] + [f'{i}_{axis}' for i in range(468) for axis in ['x', 'y', 'z']]
                writer.writerow(header)

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to grab frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    data_row = [time.time()]

                    for lm in face_landmarks.landmark:
                        x = lm.x * w
                        y = lm.y * h
                        z = lm.z * w  # Optional scaling, can also use raw z
                        data_row.extend([x, y, z])
                        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

                    # Append data row to the CSV
                    with open(data_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data_row)

            # Display annotated frame
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        st.info(f"✅ Detection stopped. Data saved to `{data_file}`")

else:
    st.warning("Click 'Start Detection' in the sidebar to begin.")

