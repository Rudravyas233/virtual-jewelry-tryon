"""
AI Virtual Jewelry Try-On
Production version optimized for Railway deployment
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from av import VideoFrame

from face_tryon import FaceTryOn
from hand_tryon import HandTryOn


# ---------------------------------------------------
# Page setup
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Virtual Jewelry Try-On",
    layout="wide"
)


# ---------------------------------------------------
# Load detectors once
# ---------------------------------------------------

@st.cache_resource
def load_detectors():
    return FaceTryOn(), HandTryOn()

face_det, hand_det = load_detectors()


# ---------------------------------------------------
# Constants
# ---------------------------------------------------

FACE_TYPES = {"Necklaces", "Earrings"}

CATEGORY_MAP = {
    "Necklaces": ("necklace", "jewelry/necklaces"),
    "Earrings": ("earrings", "jewelry/earrings"),
    "Rings": ("ring", "jewelry/rings"),
    "Bracelets": ("bracelet", "jewelry/bracelets"),
}

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def list_jewelry(folder):

    os.makedirs(folder, exist_ok=True)

    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".webp"))
    ])


def load_png(path):

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is not None and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    return img


def run_detector(
        frame,
        jtype,
        is_face,
        jewelry_img,
        v_adj,
        finger,
        ear_adj,
        ring_scale,
        bracelet_scale):

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


# ---------------------------------------------------
# WebRTC Video Processor
# ---------------------------------------------------

class JewelryProcessor(VideoProcessorBase):

    def __init__(self):

        self.jtype = None
        self.is_face = None
        self.jewelry_img = None
        self.v_adj = 40
        self.finger = "Ring"
        self.ear_adj = 0
        self.ring_scale = 1.0
        self.bracelet_scale = 1.0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # mirror camera
        img = cv2.flip(img, 1)

        # slight smoothing
        img = cv2.GaussianBlur(img, (3,3), 0)

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


# ---------------------------------------------------
# Main UI
# ---------------------------------------------------

def main():

    st.title("✨ AI Virtual Jewelry Try-On")

    mode = st.sidebar.selectbox(
        "Mode",
        ["Upload Photo", "Live Webcam"]
    )


    # ---------------------------------------------------
    # Jewelry selection
    # ---------------------------------------------------

    category = st.sidebar.selectbox(
        "Category",
        list(CATEGORY_MAP.keys())
    )

    jtype, folder = CATEGORY_MAP[category]

    is_face = category in FACE_TYPES

    items = list_jewelry(folder)

    selected_img = None

    if items:

        item = st.sidebar.selectbox(
            "Select Jewelry",
            items
        )

        path = os.path.join(folder, item)

        selected_img = load_png(path)

        st.sidebar.image(path, width=120)

    else:

        st.sidebar.warning("Add PNG jewelry images")


    # ---------------------------------------------------
    # Controls
    # ---------------------------------------------------

    v_adj = 40
    ear_adj = 0
    finger = "Ring"
    ring_scale = 1.0
    bracelet_scale = 1.0


    if category == "Necklaces":

        v_adj = st.sidebar.slider(
            "Necklace Height",
            -50,150,40
        )


    if category == "Earrings":

        ear_adj = st.sidebar.slider(
            "Earring Position",
            -80,80,0
        )


    if category == "Rings":

        finger = st.sidebar.radio(
            "Finger",
            FINGERS,
            index=3
        )

        ring_scale = st.sidebar.slider(
            "Ring Size",
            0.5,2.0,1.0
        )


    if category == "Bracelets":

        bracelet_scale = st.sidebar.slider(
            "Bracelet Size",
            0.5,2.0,1.0
        )


    # ---------------------------------------------------
    # Upload Mode
    # ---------------------------------------------------

    if mode == "Upload Photo":

        file = st.file_uploader(
            "Upload image",
            type=["jpg","jpeg","png"]
        )

        if file and selected_img is not None:

            img = Image.open(file).convert("RGB")

            frame = cv2.cvtColor(
                np.array(img),
                cv2.COLOR_RGB2BGR
            )

            result = run_detector(
                frame,
                jtype,
                is_face,
                selected_img,
                v_adj,
                finger,
                ear_adj,
                ring_scale,
                bracelet_scale
            )

            st.image(
                cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            )


    # ---------------------------------------------------
    # Webcam Mode
    # ---------------------------------------------------

    else:

        st.markdown("### Live Virtual Mirror")

        if selected_img is None:

            st.error("Select jewelry first")
            return


        rtc_config = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        })


        ctx = webrtc_streamer(

            key="jewelry",

            video_processor_factory=JewelryProcessor,

            rtc_configuration=rtc_config,

            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "frameRate": {"ideal": 30},
                },
                "audio": False
            },

            async_processing=True
        )


        if ctx.video_processor:

            ctx.video_processor.jtype = jtype
            ctx.video_processor.is_face = is_face
            ctx.video_processor.jewelry_img = selected_img
            ctx.video_processor.v_adj = v_adj
            ctx.video_processor.finger = finger
            ctx.video_processor.ear_adj = ear_adj
            ctx.video_processor.ring_scale = ring_scale
            ctx.video_processor.bracelet_scale = bracelet_scale


# ---------------------------------------------------

if __name__ == "__main__":
    main()