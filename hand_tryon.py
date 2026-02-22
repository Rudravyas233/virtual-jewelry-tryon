import cv2
import mediapipe as mp
import numpy as np
from overlay_utils import overlay_transparent

class HandTryOn:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process(self, frame, jewelry_image, jewelry_type):
        """
        Processes the frame to detect hands and overlay jewelry.
        jewelry_type: 'ring' or 'bracelet'
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return frame
        
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        h, w, _ = frame.shape
        
        if jewelry_type == 'ring':
            return self._overlay_ring(frame, hand_landmarks, jewelry_image, h, w)
        elif jewelry_type == 'bracelet':
            return self._overlay_bracelet(frame, hand_landmarks, jewelry_image, h, w)
            
        return frame

    def _overlay_ring(self, frame, landmarks, ring_img, h, w):
        # Ring finger landmarks:
        # Base: 13
        # Middle joint: 14
        
        base = landmarks[13]
        joint = landmarks[14]
        
        # Position at the center of the base segment
        ring_x = int(base.x * w)
        ring_y = int(base.y * h)
        
        # Scale based on finger segment length
        segment_len = np.sqrt((base.x - joint.x)**2 + (base.y - joint.y)**2) * h
        ring_width = int(segment_len * 1.5)
        
        # Maintain aspect ratio
        img_h, img_w = ring_img.shape[:2]
        ring_height = int((ring_width / img_w) * img_h)
        
        start_x = ring_x - ring_width // 2
        start_y = ring_y - ring_height // 2
        
        # Calculate rotation if needed (bonus)
        angle = np.degrees(np.arctan2(joint.y - base.y, joint.x - base.x)) + 90
        
        # For now, simple overlay. Full rotation logic would involve cv2.getRotationMatrix2D
        return overlay_transparent(frame, ring_img, start_x, start_y, (ring_width, ring_height))

    def _overlay_bracelet(self, frame, landmarks, bracelet_img, h, w):
        # Wrist: 0
        # Lower palm landmarks: 1, 17, 5
        
        wrist = landmarks[0]
        
        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)
        
        # Scale based on hand size (distance between wrist and index base)
        hand_scale = np.sqrt((landmarks[0].x - landmarks[5].x)**2 + (landmarks[0].y - landmarks[5].y)**2) * w
        bracelet_width = int(hand_scale * 1.2)
        
        # Maintain aspect ratio
        img_h, img_w = bracelet_img.shape[:2]
        bracelet_height = int((bracelet_width / img_w) * img_h)
        
        start_x = wrist_x - bracelet_width // 2
        start_y = wrist_y - bracelet_height // 2
        
        return overlay_transparent(frame, bracelet_img, start_x, start_y, (bracelet_width, bracelet_height))
