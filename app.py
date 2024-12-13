import cv2
import streamlit as st
from ultralytics import YOLO
st.set_page_config(page_title="PPE Detection Demo", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: black; 
            color: white;  
        }
        .title {
            text-align: center;
            color: white;  
            font-size: 36px;
            font-weight: bold;
            margin-top: 10px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: white;
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: black;
            padding: 10px 0;
        }
        .center-text {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">PPE Detection Demo</div>', unsafe_allow_html=True)
st.progress(100)

path = './model.pt'
try:
    model = YOLO(path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
video_options = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
selected_video = st.selectbox("Select a video for PPE detection:", video_options)

if selected_video:  
    st.subheader(f"Selected video: {selected_video}")
    video_path = f'./{selected_video}'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file!")
        st.stop()
    col1, col2, col3 = st.columns([1, 4, 1])  
    with col2:
        st.markdown('<div class="center-text">Processed frames with detections will be displayed here:</div>', unsafe_allow_html=True)
        processed_frame_placeholder = st.empty()

    count = 0
    while True:
        ret, frame = cap.read()
        
        # End video or error reading frame
        if not ret:
            st.info("End of video or error reading frame.")
            break
        
        count += 1
        
        # Skip frames for performance optimization
        if count % 3 != 0: 
            continue
        
        # Resize the frame to fit the display
        frame = cv2.resize(frame, (1020, 600))

        try:
            results = model(frame)
        except Exception as e:
            st.error(f"Error during model inference: {e}")
            break

        try:
            annotated_frame = results[0].plot()
        except Exception as e:
            st.warning(f"Error annotating frame: {e}")
            annotated_frame = frame
        
        # Convert frame to RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the processed frame
        processed_frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)


    cap.release()


st.markdown('<div class="footer">2024 @Copyright by Embrace AI Solutions. All Rights Reserved.</div>', unsafe_allow_html=True)
