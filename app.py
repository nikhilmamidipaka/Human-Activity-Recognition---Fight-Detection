import streamlit as st
import cv2
import numpy as np
from collections import deque
import os
import base64
#import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.models import load_model
from keras.initializers import Orthogonal

initializer = Orthogonal(gain=1.0, seed=None)
from keras.layers import Dense


# Load the model
model_file_path = "myharm.h5"  # Change this path accordingly
convlrcn_model = load_model(model_file_path)

# Define constants
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["noFights", "fights"]

def perform_action_recognition(video_file_path, output_file_path):
    video_reader = cv2.VideoCapture(video_file_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   int(video_reader.get(cv2.CAP_PROP_FPS)), (original_video_width, original_video_height))

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    while video_reader.isOpened():
        ok, frame = video_reader.read()

        if not ok:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            # Perform action recognition
            predicted_labels_probabilities = convlrcn_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

            # Draw predicted class name on frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Draw predicted class name on frame with black background box
            text_size = cv2.getTextSize(predicted_class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x, text_y = 10, 30  # Position of the text
            padding = 5  # Padding around the text
            box_coords = ((text_x, text_y + padding), (text_x + text_size[0] + padding * 2, text_y - text_size[1] - padding))

            # Draw the black background box
            cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), -1)

            # Draw the predicted class name on the frame
            cv2.putText(frame, predicted_class_name, (text_x + padding, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        video_writer.write(frame)

    video_reader.release()
    video_writer.release()

    return output_file_path

def get_binary_file_downloader_html(file_path, title="Download File"):
    with open(file_path, "rb") as f:
        video_bytes = f.read()
    b64 = base64.b64encode(video_bytes).decode()
    file_href = f'<a href="data:file/mp4;base64,{b64}" download="{os.path.basename(file_path)}">{title}</a>'
    return file_href

def main():
    st.title("Human Activity Prediction")

    uploaded_file = st.file_uploader("Upload a video", type=['mp4'])
    if uploaded_file is not None:
        test_videos_directory = 'test_videos'
        os.makedirs(test_videos_directory, exist_ok=True)
        video_file_path = os.path.join(test_videos_directory, uploaded_file.name)
        with open(video_file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"You uploaded: {uploaded_file.name}")

        output_video_file_path = f'{test_videos_directory}/{uploaded_file.name}-Output-SeqLen{SEQUENCE_LENGTH}.mp4'
        output_video_file_path = perform_action_recognition(video_file_path, output_video_file_path)

        st.success("Prediction complete! You can download the output video below.")
        st.markdown(get_binary_file_downloader_html(output_video_file_path, "Download Predicted Video"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
