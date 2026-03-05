"""
app.py — AI Virtual Jewelry Try-On
Stable production version
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time

from face_tryon import FaceTryOn
from hand_tryon import HandTryOn


# -----------------------------------------------------------
# Page Config
# -----------------------------------------------------------

st.set_page_config(
    page_title="AI Virtual Jewelry Try-On",
    layout="wide",
)

# -----------------------------------------------------------
# Cache detectors
# -----------------------------------------------------------

@st.cache_resource
def load_detectors():
    return FaceTryOn(), HandTryOn()

face_det, hand_det = load_detectors()

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------

FACE_TYPES = {"Necklaces", "Earrings"}

CATEGORY_MAP = {
    "Necklaces": ("necklace", "jewelry/necklaces"),
    "Earrings": ("earrings", "jewelry/earrings"),
    "Rings": ("ring", "jewelry/rings"),
    "Bracelets": ("bracelet", "jewelry/bracelets"),
}

FINGER_OPTS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------

def list_jewelry(folder):

    os.makedirs(folder, exist_ok=True)

    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".webp"))
    )


def load_img(path):

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is not None and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    return img


def run_detector(frame, jtype, is_face, jewelry_img,
                 v_adj, finger, ear_adj,
                 ring_scale, bracelet_scale):

    if is_face:
        return face_det.process(
            frame,
            jewelry_img,
            jtype,
            v_offset=v_adj,
            ear_v_offset=ear_adj
        )

    return hand_det.process(
        frame,
        jewelry_img,
        jtype,
        finger=finger,
        ring_scale=ring_scale,
        bracelet_scale=bracelet_scale
    )


# -----------------------------------------------------------
# Main UI
# -----------------------------------------------------------

def main():

    st.sidebar.title("💎 Virtual Jewelry Try-On")

    mode = st.sidebar.selectbox(
        "Mode",
        ["Upload Photo", "Live Webcam"]
    )

    st.sidebar.markdown("---")

    category = st.sidebar.selectbox(
        "Category",
        list(CATEGORY_MAP.keys())
    )

    jtype, folder = CATEGORY_MAP[category]

    is_face = category in FACE_TYPES

    items = list_jewelry(folder)

    selected_img = None

    if items:

        choice = st.sidebar.selectbox(
            "Select Jewelry",
            items
        )

        path = os.path.join(folder, choice)

        selected_img = load_img(path)

        st.sidebar.image(
            Image.open(path).convert("RGBA"),
            width=140
        )

    else:

        st.sidebar.warning(
            f"No PNG files found in `{folder}`"
        )

    # -------------------------------------------------------
    # Controls
    # -------------------------------------------------------

    st.sidebar.markdown("---")

    st.sidebar.subheader("Fine Adjust")

    v_adj = 40
    ear_adj = 0
    ring_scale = 1.0
    bracelet_scale = 1.0
    selected_finger = "Ring"

    if is_face and category == "Necklaces":

        v_adj = st.sidebar.slider(
            "Necklace Height",
            -50, 150, 40
        )

    if is_face and category == "Earrings":

        ear_adj = st.sidebar.slider(
            "Earring Offset",
            -80, 80, 0
        )

    if category == "Rings":

        selected_finger = st.sidebar.radio(
            "Finger",
            FINGER_OPTS,
            index=3
        )

        ring_scale = st.sidebar.slider(
            "Ring Size",
            0.5, 2.0, 1.0
        )

    if category == "Bracelets":

        bracelet_scale = st.sidebar.slider(
            "Bracelet Size",
            0.5, 2.0, 1.0
        )

    # -------------------------------------------------------
    # Main Layout
    # -------------------------------------------------------

    st.title("✨ AI Virtual Jewelry Try-On")

    col_view, col_info = st.columns([3, 1])

    with col_info:

        st.markdown(
            """
**Tips for best results**

• Good lighting  
• Jewelry PNG with transparent background  
• Keep hand steady  
• Webcam works best
"""
        )

    # -------------------------------------------------------
    # Upload Mode
    # -------------------------------------------------------

    if mode == "Upload Photo":

        up = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"]
        )

        if up and selected_img is not None:

            pil = Image.open(up).convert("RGB")

            bgr = cv2.cvtColor(
                np.array(pil),
                cv2.COLOR_RGB2BGR
            )

            with st.spinner("Processing..."):

                out = run_detector(
                    bgr.copy(),
                    jtype,
                    is_face,
                    selected_img,
                    v_adj,
                    selected_finger,
                    ear_adj,
                    ring_scale,
                    bracelet_scale
                )

            with col_view:

                st.image(
                    cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                    use_column_width=True
                )

            if st.button("Save Image"):

                fname = f"tryon_{int(time.time())}.png"

                cv2.imwrite(fname, out)

                st.success(f"Saved: {fname}")

    # -------------------------------------------------------
    # Webcam Mode
    # -------------------------------------------------------

    else:

        st.subheader("Live Virtual Mirror")

        run = st.checkbox("Start Camera")

        frame_placeholder = st.empty()

        if run and selected_img is not None:

            cap = cv2.VideoCapture(0)

            # Force HD
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            while run:

                ret, frame = cap.read()

                if not ret:
                    st.error("Camera error")
                    break

                frame = cv2.flip(frame, 1)

                out = run_detector(
                    frame,
                    jtype,
                    is_face,
                    selected_img,
                    v_adj,
                    selected_finger,
                    ear_adj,
                    ring_scale,
                    bracelet_scale
                )

                frame_placeholder.image(
                    cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_column_width=True
                )

            cap.release()

        elif run and selected_img is None:

            st.error("Select jewelry first")


# -----------------------------------------------------------
# Run
# -----------------------------------------------------------

if __name__ == "__main__":
    main()