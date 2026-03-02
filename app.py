"""
app.py — AI Virtual Boutique (Production)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time

from face_tryon import FaceTryOn
from hand_tryon import HandTryOn

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Virtual Jewelry Try-On",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0a0c0f; color: #f0f0f0; }
section[data-testid="stSidebar"] {
    background: #111318; border-right: 1px solid #2a2d36;
}
h1 { font-family: 'Playfair Display', serif;
     background: linear-gradient(90deg, #D4AF37, #F8E77C, #D4AF37);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stSelectbox label, .stCheckbox label, .stSlider label,
.stFileUploader label, .stRadio label {
    color: #c8c8c8 !important; font-size: 0.85rem;
}
div.stButton > button {
    background: linear-gradient(135deg, #D4AF37, #8B6914);
    color: #000; font-weight: 700; border-radius: 8px;
    border: none; width: 100%; padding: 0.5rem;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; }
.info-card {
    background: #161a22; border: 1px solid #2a2d36;
    border-radius: 10px; padding: 16px;
    font-size: 0.82rem; line-height: 1.7;
}
.tip { color: #D4AF37; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_detectors():
    return FaceTryOn(), HandTryOn()


face_det, hand_det = get_detectors()

FACE_TYPES   = {"Necklaces", "Earrings"}
CATEGORY_MAP = {
    "Necklaces": ("necklace",  "jewelry/necklaces"),
    "Earrings":  ("earrings",  "jewelry/earrings"),
    "Rings":     ("ring",      "jewelry/rings"),
    "Bracelets": ("bracelet",  "jewelry/bracelets"),
}
FINGER_OPTS  = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def list_jewelry(folder):
    os.makedirs(folder, exist_ok=True)
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith((".png", ".webp")))


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def run_detector(frame, jtype, is_face, jewelry_img, v_adj, finger,
                 ear_adj=0, ring_scale=1.0, bracelet_scale=1.0):
    if is_face:
        return face_det.process(frame, jewelry_img, jtype,
                                v_offset=v_adj, ear_v_offset=ear_adj)
    return hand_det.process(frame, jewelry_img, jtype, finger=finger,
                            ring_scale=ring_scale, bracelet_scale=bracelet_scale)




# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ─────────────────────────────────────────────────────────────────
    st.sidebar.markdown("## ✦ Virtual Boutique")
    mode = st.sidebar.selectbox("Mode", ["📷  Upload Photo", "🎥  Live Webcam"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Collection")
    category         = st.sidebar.selectbox("Category", list(CATEGORY_MAP.keys()))
    jtype, folder    = CATEGORY_MAP[category]
    is_face          = category in FACE_TYPES
    items            = list_jewelry(folder)

    selected_img = None
    if items:
        choice       = st.sidebar.selectbox(f"Select {category[:-1]}", items)
        path         = os.path.join(folder, choice)
        selected_img = load_img(path)
        try:
            st.sidebar.image(Image.open(path).convert("RGBA"),
                             width=140, caption=choice)
        except Exception:
            pass
    else:
        st.sidebar.warning(
            f"No PNG files in `{folder}/`.\n\n"
            "Add transparent PNG jewelry images to use the try-on."
        )

    # ── Fine-tuning controls ────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Fine‑Tuning")

    v_adj           = 40
    selected_finger = "Ring"
    ear_adj         = 0
    ring_scale      = 1.0
    bracelet_scale  = 1.0

    if is_face and category == "Necklaces":
        v_adj = st.sidebar.slider("Necklace Height ↕", -50, 150, 40,
                                  help="– = higher,  + = lower")

    if is_face and category == "Earrings":
        ear_adj = st.sidebar.slider(
            "Earring Position ↕", -80, 80, 0,
            help="Slide to move earrings up (–) or down (+)"
        )

    if category == "Rings":
        st.sidebar.markdown("#### 💍 Choose Finger")
        selected_finger = st.sidebar.radio(
            "", FINGER_OPTS, index=FINGER_OPTS.index("Ring"),
            horizontal=True
        )
        st.sidebar.caption(
            f"Ring will be placed on your **{selected_finger}** finger."
        )
        ring_scale = st.sidebar.slider(
            "Ring Size 🔍", 0.5, 2.0, 1.0, step=0.05,
            help="Adjust ring size to fit your finger tightly"
        )

    if category == "Bracelets":
        bracelet_scale = st.sidebar.slider(
            "Bracelet Size 🔍", 0.5, 2.0, 1.0, step=0.05,
            help="Adjust bracelet width to fit your wrist"
        )


    # ── Main panel ───────────────────────────────────────────────────────────────
    st.title("✨  AI Virtual Jewelry Try-On")
    col_view, col_info = st.columns([3, 1], gap="large")

    with col_info:
        st.markdown(
            '<div class="info-card">'
            '<span class="tip">How to get best results</span><br><br>'
            '🔆 <b>Good lighting</b> on face / hand<br>'
            '📐 Jewelry PNG should be <b>centred</b> with <b>transparent</b> BG<br>'
            '🤚 For rings — <b>palm facing camera</b><br>'
            '📏 Move <b>slowly</b> for stable tracking<br>'
            '💎 <b>Webcam</b> mode gives real-time try-on<br><br>'
            '<i>Powered by MediaPipe World Landmarks<br>'
            '+ OpenCV Perspective Transform</i>'
            '</div>',
            unsafe_allow_html=True
        )

    with col_view:
        # ── Upload ───────────────────────────────────────────────────────────────
        if "Upload" in mode:
            up = st.file_uploader("Upload portrait or hand photo",
                                  type=["jpg", "jpeg", "png"])
            if up and selected_img is not None:
                pil = Image.open(up).convert("RGB")
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                with st.spinner("Detecting landmarks & rendering…"):
                    out = run_detector(bgr.copy(), jtype, is_face,
                                       selected_img, v_adj, selected_finger,
                                       ear_adj, ring_scale, bracelet_scale)
                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                         use_container_width=True,
                         caption="Virtual Try-On Result")
                if st.button("💾  Save Image"):
                    fname = f"tryon_{int(time.time())}.png"
                    cv2.imwrite(fname, out)
                    st.success(f"Saved → {fname}")
            elif not items:
                st.info("Add jewelry PNG files to start.")

        # ── Webcam ───────────────────────────────────────────────────────────────
        else:
            st.markdown("### 📽️  Live Virtual Mirror")
            run    = st.checkbox("▶ Activate Camera", value=False)
            frame_holder = st.empty()
            fps_holder   = st.empty()

            if run and selected_img is not None:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Cannot access camera. Check permissions.")
                else:
                    prev_t = time.time()
                    while run:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Camera read failed.")
                            break
                        frame = cv2.flip(frame, 1)
                        out   = run_detector(frame, jtype, is_face,
                                             selected_img, v_adj, selected_finger,
                                             ear_adj, ring_scale, bracelet_scale)
                        frame_holder.image(
                            cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                            channels="RGB", use_container_width=True
                        )
                        now   = time.time()
                        fps   = 1.0 / max(now - prev_t, 1e-9)
                        prev_t = now
                        fps_holder.caption(f"⚡ {fps:.1f} FPS")
                    cap.release()
            elif run and selected_img is None:
                st.error("Select a jewelry item from the sidebar first.")


if __name__ == "__main__":
    main()
