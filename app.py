import streamlit as st
import requests


from PIL import Image

def resize_image(image_path, target_size):
    img = Image.open(image_path)
    img_resized = img.resize(target_size)
    return img_resized
def get_google_drive_direct_link(gdrive_link):
    # Extract the file ID from the Google Drive link
    file_id = gdrive_link.split("/")[-2]
    
    # Make a request to Google Drive API to get the direct link
    response = requests.get(f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media", allow_redirects=True)
    
    # Return the direct link
    return response.url

def main():
    
    # Title and Header
    st.title("Fight Detection: Human Activity Recognition")
    st.markdown("---")
    
    # Introduction
    st.write("Welcome to Fight Detection, an application for recognizing aggressive human activities in videos.")
    st.write("Our model uses deep learning techniques to analyze video footage and identify instances of fights.")
    st.write("With real-time detection capabilities, our application can help monitor and respond to potential security threats.")
    
    # Features Section
    st.markdown("## Features")
    st.write("1. **Real-Time Detection:** Detect fights as they happen in live video streams.")
    st.write("2. **Accuracy:** Our model achieves high accuracy in identifying fight scenes.")
    st.write("3. **Customization:** Fine-tune the model for specific environments or scenarios.")

    st.markdown("## About the LRCN Model")
    st.write("The Long-term Recurrent Convolutional Network (LRCN) is a hybrid deep learning model that combines Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs).")
    st.write("It excels in tasks involving sequential data like video analysis, where both spatial and temporal features are crucial.")
    st.write("In the context of this project, the LRCN model accurately identifies aggressive human activities in video footage by capturing both spatial and temporal characteristics, making it a powerful tool for such tasks.")
    st.write("Key Components:")
    st.write("- CNNs: Extract spatial features from video frames.")
    st.write("- RNNs (LSTM/GRU): Model temporal dependencies across frames.")
    st.write("- Integration: Combines spatial and temporal information.")
    st.write("- Training: Trained via backpropagation, fine-tuning for specific datasets.")
    st.write("- Applications: Used in action recognition, activity recognition, and video captioning.")
    
    
    st.markdown("## Demo Images")
    st.write("See examples of 'Fights' and 'No Fights' scenarios:")
    
    # Load and display the images
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Fights**")
        fight_image_resized = resize_image("fight.jfif", (300, 300))
        st.image(fight_image_resized, use_column_width=True)
        
    with col2:
        st.write("**No Fights**")
        no_fight_image_resized = resize_image("no fight.jfif", (300, 300))
        st.image(no_fight_image_resized, use_column_width=True)
    
    # About Us Section
    st.markdown("## About Us")
    st.write("We are a team of passionate developers dedicated to enhancing security and safety through technology.")
    st.write("Our mission is to leverage machine learning and computer vision to create innovative solutions for real-world problems.")
    
    
if __name__ == "__main__":
    main()
