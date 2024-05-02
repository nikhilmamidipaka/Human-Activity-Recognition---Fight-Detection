import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
from collections import deque
import os
import base64
from tensorflow.keras.models import load_model

# Load the model
model_file_path = "myharm.h5"  # Change this path accordingly
convlrcn_model = load_model(model_file_path)

# Define constants
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["noFights", "fights"]

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        resized_frame = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        self.frames_queue.append(normalized_frame)

        if len(self.frames_queue) == SEQUENCE_LENGTH:
            predicted_class_name = self.predict_activity(self.frames_queue)

            # Draw predicted class name on frame
            cv2.putText(img, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

    def predict_activity(self, frames_queue):
        # Perform action recognition using the loaded model
        predicted_labels_probabilities = convlrcn_model.predict(np.expand_dims(frames_queue, axis=0))[0]
        predicted_label = np.argmax(predicted_labels_probabilities)
        predicted_class_name = CLASSES_LIST[predicted_label]
        return predicted_class_name

def main():
    st.title("Human Activity Prediction")

    webrtc_streamer(
    key="example",
    video_processor_factory=VideoTransformer,
    mode=WebRtcMode.SENDRECV, rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
)

if __name__ == "__main__":
    main()
