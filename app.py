import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from face_tryon import FaceTryOn
from hand_tryon import HandTryOn

# Page Config
st.set_page_config(page_title="AI Virtual Jewelry Try-On", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSelectbox label, .stHeader {
        color: #f0f2f6 !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Try-On Modules
@st.cache_resource
def get_detectors():
    return FaceTryOn(), HandTryOn()

face_detector, hand_detector = get_detectors()

def load_jewelry_images(category):
    path = f"jewelry/{category}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    images = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return images

def main():
    st.title("✨ Virtual Jewelry Try-On")
    st.sidebar.title("Configuration")
    
    mode = st.sidebar.selectbox("Choose Mode", ["Upload Image", "Live Webcam"])
    
    category = st.sidebar.selectbox("Jewelry Category", ["Necklaces", "Earrings", "Rings", "Bracelets"])
    jewelry_items = load_jewelry_images(category.lower())
    
    selected_jewelry = None
    if jewelry_items:
        jewelry_choice = st.sidebar.selectbox(f"Select {category}", jewelry_items)
        jewelry_path = os.path.join("jewelry", category.lower(), jewelry_choice)
        selected_jewelry = cv2.imread(jewelry_path, cv2.IMREAD_UNCHANGED)
    else:
        st.sidebar.warning(f"No {category} found. Please add PNG images to `jewelry/{category.lower()}/`")

    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file and selected_jewelry is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            processed_img = img_bgr.copy()
            if category in ["Necklaces", "Earrings"]:
                processed_img = face_detector.process(processed_img, selected_jewelry, category[:-1].lower())
            else:
                processed_img = hand_detector.process(processed_img, selected_jewelry, category[:-1].lower())
                
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Save button
            if st.button("Save Result"):
                res_path = "output_tryon.png"
                cv2.imwrite(res_path, processed_img)
                st.success(f"Image saved as {res_path}")

    elif mode == "Live Webcam":
        st.info("Starting Webcam... Please ensure your camera is accessible.")
        run_webcam = st.checkbox("Run Webcam")
        FRAME_WINDOW = st.image([])
        
        if run_webcam and selected_jewelry is not None:
            cap = cv2.VideoCapture(0)
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam.")
                    break
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                
                # Process
                if category in ["Necklaces", "Earrings"]:
                    processed_frame = face_detector.process(frame, selected_jewelry, category[:-1].lower())
                else:
                    processed_frame = hand_detector.process(frame, selected_jewelry, category[:-1].lower())
                
                FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            
            cap.release()
        elif selected_jewelry is None:
            st.error("Please select a jewelry item first.")

if __name__ == "__main__":
    main()
