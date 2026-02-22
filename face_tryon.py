import cv2
import mediapipe as mp
import numpy as np
from overlay_utils import overlay_transparent

class FaceTryOn:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process(self, frame, jewelry_image, jewelry_type):
        """
        Processes the frame to detect face and overlay jewelry.
        jewelry_type: 'necklace' or 'earrings'
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return frame
        
        face_landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        
        if jewelry_type == 'necklace':
            return self._overlay_necklace(frame, face_landmarks, jewelry_image, h, w)
        elif jewelry_type == 'earrings':
            return self._overlay_earrings(frame, face_landmarks, jewelry_image, h, w)
            
        return frame

    def _overlay_necklace(self, frame, landmarks, necklace_img, h, w):
        # Key landmarks for necklace:
        # Chin: 152
        # Left Jaw: 234
        # Right Jaw: 454
        
        chin = landmarks[152]
        l_jaw = landmarks[234]
        r_jaw = landmarks[454]
        
        # Calculate neck position (slightly below chin)
        neck_x = int(chin.x * w)
        neck_y = int(chin.y * h) + int(0.1 * h) # Offset below chin
        
        # Calculate width based on jaw distance
        jaw_dist = np.sqrt((l_jaw.x - r_jaw.x)**2 + (l_jaw.y - r_jaw.y)**2) * w
        necklace_width = int(jaw_dist * 1.5) # Scale factor
        
        # Maintain aspect ratio
        img_h, img_w = necklace_img.shape[:2]
        necklace_height = int((necklace_width / img_w) * img_h)
        
        # Center the necklace
        start_x = neck_x - necklace_width // 2
        start_y = neck_y - necklace_height // 4 # Adjust vertical centering
        
        return overlay_transparent(frame, necklace_img, start_x, start_y, (necklace_width, necklace_height))

    def _overlay_earrings(self, frame, landmarks, earring_img, h, w):
        # Ear regions:
        # Left: 234 (jaw) -> approx earring location below 234/127
        # Right: 454 (jaw) -> approx earring location below 454/356
        
        l_ear = landmarks[234]
        r_ear = landmarks[454]
        
        # Scale based on face size
        face_width = np.sqrt((landmarks[234].x - landmarks[454].x)**2 + (landmarks[234].y - landmarks[454].y)**2) * w
        earring_size = int(face_width * 0.2)
        
        # Maintain aspect ratio
        img_h, img_w = earring_img.shape[:2]
        earring_h = int((earring_size / img_w) * img_h)
        
        # Left Earring
        lx = int(l_ear.x * w) - earring_size // 2
        ly = int(l_ear.y * h) + int(0.02 * h)
        frame = overlay_transparent(frame, earring_img, lx, ly, (earring_size, earring_h))
        
        # Right Earring (Note: could flip image for realism)
        rx = int(r_ear.x * w) - earring_size // 2
        ry = int(r_ear.y * h) + int(0.02 * h)
        frame = overlay_transparent(frame, earring_img, rx, ry, (earring_size, earring_h))
        
        return frame
