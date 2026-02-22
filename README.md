# AI-Powered Virtual Jewelry Try-On System

A production-ready AI-powered virtual jewelry try-on web application using MediaPipe for real-time face and hand landmark detection and OpenCV for seamless jewelry overlay.

## Features
- **Input Modes**: Upload Image or Live Webcam.
- **Jewelry Support**: Necklaces, Earrings, Rings, and Bracelets.
- **Accurate Tracking**: Uses MediaPipe FaceMesh and Hands for precise placement.
- **Premium UI**: Built with Streamlit for a sleek, responsive experience.

## Project Structure
```
virtual_tryon/
├── app.py              # Main Streamlit application
├── overlay_utils.py    # Alpha blending and overlay logic
├── face_tryon.py       # Face detection & Necklace/Earring logic
├── hand_tryon.py       # Hand detection & Ring/Bracelet logic
├── jewelry/            # Folder for jewelry PNG assets
│   ├── necklaces/
│   ├── earrings/
│   ├── rings/
│   └── bracelets/
├── requirements.txt    # Python dependencies
└── README.md           # Instructions
```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Jewelry Assets**:
   - Place transparent PNG images of jewelry in their respective folders under `jewelry/`.
   - Ensure images are centered and high resolution.

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Technical Details
- **MediaPipe**: Used for extracting 468 3D face landmarks and 21 hand landmarks.
- **OpenCV**: Handles image transformations, resizing, and alpha blending for realistic overlays.
- **Streamlit**: Provides the interactive web interface.

## Future Improvements
- 3D Jewelry rendering.
- Realistic lighting and shadow effects.
- Multi-person support.
