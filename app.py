"""
app.py — AI Virtual Jewelry Try-On
Cloud compatible version (Railway / Streamlit)
Uses browser webcam via streamlit-webrtc
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from face_tryon import FaceTryOn
from hand_tryon import HandTryOn


# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------

st.set_page_config(
    page_title="AI Virtual Jewelry Try-On",
    layout="wide"
)


# ------------------------------------------------------------
# Load Detectors (cached)
# ------------------------------------------------------------

@st.cache_resource
def load_detectors():
    return FaceTryOn(), HandTryOn()


face_det, hand_det = load_detectors()


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

FACE_TYPES = {"Necklaces", "Earrings"}

CATEGORY_MAP = {
    "Necklaces": ("necklace", "jewelry/necklaces"),
    "Earrings": ("earrings", "jewelry/earrings"),
    "Rings": ("ring", "jewelry/rings"),
    "Bracelets": ("bracelet", "jewelry/bracelets"),
}

FINGER_OPTS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

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

    else:

        return hand_det.process(
            frame,
            jewelry_img,
            jtype,
            finger=finger,
            ring_scale=ring_scale,
            bracelet_scale=bracelet_scale
        )


# ------------------------------------------------------------
# Webcam Processor
# ------------------------------------------------------------

class VideoProcessor(VideoTransformerBase):

    def __init__(self):

        self.jtype = None
        self.is_face = False
        self.jewelry_img = None
        self.v_adj = 0
        self.finger = "Ring"
        self.ear_adj = 0
        self.ring_scale = 1.0
        self.bracelet_scale = 1.0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        if self.jewelry_img is None:
            return img

        out = run_detector(
            img,
            self.jtype,
            self.is_face,
            self.jewelry_img,
            self.v_adj,
            self.finger,
            self.ear_adj,
            self.ring_scale,
            self.bracelet_scale
        )

        return out


# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------

def main():

    st.title("✨ AI Virtual Jewelry Try-On")

    st.sidebar.title("💎 Controls")

    mode = st.sidebar.selectbox(
        "Mode",
        ["Upload Photo", "Live Webcam"]
    )

    category = st.sidebar.selectbox(
        "Jewelry Category",
        list(CATEGORY_MAP.keys())
    )

    jtype, folder = CATEGORY_MAP[category]

    is_face = category in FACE_TYPES

    items = list_jewelry(folder)

    selected_img = None

    if items:

        choice = st.sidebar.selectbox("Select Jewelry", items)

        path = os.path.join(folder, choice)

        selected_img = load_img(path)

        st.sidebar.image(
            Image.open(path).convert("RGBA"),
            width=150
        )

    else:

        st.sidebar.warning(f"No images in {folder}")


    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    v_adj = 40
    ear_adj = 0
    ring_scale = 1.0
    bracelet_scale = 1.0
    selected_finger = "Ring"

    st.sidebar.markdown("---")

    if category == "Necklaces":

        v_adj = st.sidebar.slider(
            "Necklace Height",
            -50, 150, 40
        )

    if category == "Earrings":

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


    # --------------------------------------------------------
    # Upload Mode
    # --------------------------------------------------------

    if mode == "Upload Photo":

        uploaded = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded and selected_img is not None:

            pil = Image.open(uploaded).convert("RGB")

            frame = cv2.cvtColor(
                np.array(pil),
                cv2.COLOR_RGB2BGR
            )

            with st.spinner("Processing..."):

                out = run_detector(
                    frame.copy(),
                    jtype,
                    is_face,
                    selected_img,
                    v_adj,
                    selected_finger,
                    ear_adj,
                    ring_scale,
                    bracelet_scale
                )

            st.image(
                cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

            if st.button("Save Result"):

                fname = f"tryon_{int(time.time())}.png"

                cv2.imwrite(fname, out)

                st.success(f"Saved: {fname}")


    # --------------------------------------------------------
    # Webcam Mode (Browser Camera)
    # --------------------------------------------------------

    else:

        st.subheader("📷 Live Virtual Mirror")

        processor = VideoProcessor()

        processor.jtype = jtype
        processor.is_face = is_face
        processor.jewelry_img = selected_img
        processor.v_adj = v_adj
        processor.finger = selected_finger
        processor.ear_adj = ear_adj
        processor.ring_scale = ring_scale
        processor.bracelet_scale = bracelet_scale

        webrtc_streamer(
            key="jewelry",
            video_processor_factory=lambda: processor,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True,
        )


# ------------------------------------------------------------
# Run App
# ------------------------------------------------------------

if __name__ == "__main__":
    main()