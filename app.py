"""
app.py — AI Virtual Boutique (Production + Smooth WebRTC)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from av import VideoFrame

from face_tryon import FaceTryOn
from hand_tryon import HandTryOn


# ─────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Virtual Jewelry Try-On",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}
.main {
    background: #0a0c0f;
    color: #f0f0f0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# Load detectors once
# ─────────────────────────────────────────────────────

@st.cache_resource
def get_detectors():
    return FaceTryOn(), HandTryOn()

face_det, hand_det = get_detectors()


# ─────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────

FACE_TYPES = {"Necklaces", "Earrings"}

CATEGORY_MAP = {
    "Necklaces": ("necklace", "jewelry/necklaces"),
    "Earrings": ("earrings", "jewelry/earrings"),
    "Rings": ("ring", "jewelry/rings"),
    "Bracelets": ("bracelet", "jewelry/bracelets"),
}

FINGER_OPTS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

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
                 v_adj, finger,
                 ear_adj=0,
                 ring_scale=1.0,
                 bracelet_scale=1.0):

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


# ─────────────────────────────────────────────────────
# WebRTC Video Processor
# ─────────────────────────────────────────────────────

class JewelryVideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.jtype = None
        self.is_face = None
        self.jewelry_img = None
        self.v_adj = 0
        self.finger = "Ring"
        self.ear_adj = 0
        self.ring_scale = 1.0
        self.bracelet_scale = 1.0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

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

        return VideoFrame.from_ndarray(out, format="bgr24")


# ─────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────

def main():

    st.sidebar.title("✦ Virtual Boutique")

    mode = st.sidebar.selectbox(
        "Mode",
        ["📷 Upload Photo", "🎥 Live Webcam"]
    )

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
            "Select Item",
            items
        )

        path = os.path.join(folder, choice)

        selected_img = load_img(path)

        try:
            st.sidebar.image(
                Image.open(path).convert("RGBA"),
                width=140
            )
        except:
            pass

    else:

        st.sidebar.warning(
            f"No PNG files found in `{folder}`"
        )


    # ─────────────────────────────────
    # Fine Controls
    # ─────────────────────────────────

    v_adj = 40
    selected_finger = "Ring"
    ear_adj = 0
    ring_scale = 1.0
    bracelet_scale = 1.0

    if is_face and category == "Necklaces":

        v_adj = st.sidebar.slider(
            "Necklace Height",
            -50, 150, 40
        )

    if is_face and category == "Earrings":

        ear_adj = st.sidebar.slider(
            "Earring Position",
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
            0.5, 2.0, 1.0, step=0.05
        )

    if category == "Bracelets":

        bracelet_scale = st.sidebar.slider(
            "Bracelet Size",
            0.5, 2.0, 1.0, step=0.05
        )


    # ─────────────────────────────────
    # Title
    # ─────────────────────────────────

    st.title("✨ AI Virtual Jewelry Try-On")


    # ─────────────────────────────────
    # Upload Mode
    # ─────────────────────────────────

    if "Upload" in mode:

        up = st.file_uploader(
            "Upload portrait or hand photo",
            type=["jpg", "jpeg", "png"]
        )

        if up and selected_img is not None:

            pil = Image.open(up).convert("RGB")

            bgr = cv2.cvtColor(
                np.array(pil),
                cv2.COLOR_RGB2BGR
            )

            with st.spinner("Rendering..."):

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

            st.image(
                cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )


    # ─────────────────────────────────
    # Webcam Mode
    # ─────────────────────────────────

    else:

        st.markdown("### 📽️ Live Virtual Mirror")

        if selected_img is None:

            st.error("Select jewelry first")
            return

        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        })

        ctx = webrtc_streamer(

            key="jewelry-tryon",

            video_processor_factory=JewelryVideoProcessor,

            rtc_configuration=RTC_CONFIGURATION,

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True
        )

        if ctx.video_processor:

            ctx.video_processor.jtype = jtype
            ctx.video_processor.is_face = is_face
            ctx.video_processor.jewelry_img = selected_img
            ctx.video_processor.v_adj = v_adj
            ctx.video_processor.finger = selected_finger
            ctx.video_processor.ear_adj = ear_adj
            ctx.video_processor.ring_scale = ring_scale
            ctx.video_processor.bracelet_scale = bracelet_scale


# ─────────────────────────────────

if __name__ == "__main__":
    main()