"""
face_tryon.py
=============
Production necklace & earring placement using FaceMesh.

EARRING geometry (correct):
  • Anchor = LM 234 (left) / LM 454 (right) — jaw-cheek junction,
    the lowest lateral silhouette points ≈ earlobe level.
  • Push = PURELY HORIZONTAL (±x only) — no diagonal y component.
  • earring_center_y = lobe_y + ear_drop/2
    → top of earring sits at the lobe, pendant hangs downward.
"""

import cv2
import mediapipe as mp
import numpy as np

from overlay_utils import alpha_blend
from transform_utils import (
    calculate_distance, calculate_angle, unit_vector_2d,
    resize_to_width, rotate_image, ParamSmoother
)


class FaceTryOn:
    NOSE_TIP  = 1
    CHIN      = 152

    # Jaw-cheek boundary (used for face width measurement)
    L_JAW     = 234
    R_JAW     = 454

    # True earlobe landmarks in MediaPipe FaceMesh 468-point model
    L_LOBE    = 177   # left  earlobe
    R_LOBE    = 401   # right earlobe

    # Ear upper reference for tilt angle
    L_EAR_UP  = 127
    R_EAR_UP  = 356

    def __init__(self):
        mp_fm = mp.solutions.face_mesh
        self.face_mesh = mp_fm.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.necklace_sm = ParamSmoother(0.72)
        self.ear_l_sm    = ParamSmoother(0.72)
        self.ear_r_sm    = ParamSmoother(0.72)

    # ── Public ──────────────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray,
                jewelry_img: np.ndarray,
                jewelry_type: str,
                v_offset: int = 40,
                ear_v_offset: int = 0) -> np.ndarray:
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return frame
        lm = res.multi_face_landmarks[0].landmark

        if jewelry_type == 'necklace':
            return self._apply_necklace(frame, lm, jewelry_img, w, h, v_offset)
        if jewelry_type == 'earrings':
            return self._apply_earrings(frame, lm, jewelry_img, w, h, ear_v_offset)
        return frame

    def _px(self, lm, idx, w, h):
        l = lm[idx]
        return np.array([l.x * w, l.y * h, l.z])

    # ── NECKLACE ─────────────────────────────────────────────────────────────────
    def _apply_necklace(self, frame, lm, img, w, h, v_offset):
        p_ljaw = self._px(lm, self.L_JAW,    w, h)
        p_rjaw = self._px(lm, self.R_JAW,    w, h)
        p_chin = self._px(lm, self.CHIN,     w, h)
        p_nose = self._px(lm, self.NOSE_TIP, w, h)

        face_h    = abs(float(p_chin[1]) - float(p_nose[1]))
        jaw_dist  = calculate_distance(p_ljaw[:2], p_rjaw[:2])

        cx_raw    = float((p_ljaw[0] + p_rjaw[0]) / 2.0)
        cy_raw    = float(p_chin[1]) + face_h * (1.10 + v_offset / 200.0)

        z_avg     = float((p_ljaw[2] + p_rjaw[2]) / 2.0)
        z_factor  = max(0.7, min(1.3, 1.0 - z_avg * 1.5))
        scale_raw = jaw_dist * 1.20 * z_factor
        angle_raw = -calculate_angle(p_ljaw[:2], p_rjaw[:2])

        sm = self.necklace_sm.smooth(cx=cx_raw, cy=cy_raw,
                                     scale=scale_raw, angle=angle_raw)
        nw = max(int(sm['scale']), 10)
        neck_img = resize_to_width(img, nw)
        neck_img = rotate_image(neck_img, float(sm['angle']))
        return alpha_blend(frame, neck_img, float(sm['cx']), float(sm['cy']))

    # ── EARRINGS ─────────────────────────────────────────────────────────────────
    def _apply_earrings(self, frame, lm, img, w, h, ear_v_offset=0):
        """
        CORRECT earring placement strategy:

        ANCHOR:
          • Use LM 177 (left earlobe) and LM 401 (right earlobe) directly.
            These are the TRUE earlobe landmark points in MediaPipe FaceMesh.
          • X is read directly from the lobe landmark — no extra push needed.
          • A small outward nudge (5% of face_w) keeps earring from overlapping hair.

        VERTICAL:
          • lobe landmark Y = top attachment point of earring.
          • Earring center placed at lobe_y + ear_drop/2
            → top of earring sits at the lobe, pendant hangs naturally downward.

        SIZE:
          • ear_sz = 22% of jaw-width (face_w) — visible but proportionate.
        """
        p_nose    = self._px(lm, self.NOSE_TIP, w, h)
        p_chin    = self._px(lm, self.CHIN,     w, h)

        # True earlobe positions
        p_l_lobe  = self._px(lm, self.L_LOBE,    w, h)   # LM 177
        p_r_lobe  = self._px(lm, self.R_LOBE,    w, h)   # LM 401

        # Jaw width for sizing reference
        p_l_jaw   = self._px(lm, self.L_JAW,    w, h)
        p_r_jaw   = self._px(lm, self.R_JAW,    w, h)

        # For tilt angle
        p_l_up    = self._px(lm, self.L_EAR_UP, w, h)   # LM 127
        p_r_up    = self._px(lm, self.R_EAR_UP, w, h)   # LM 356

        face_w    = calculate_distance(p_l_jaw[:2], p_r_jaw[:2])

        # Earring image dimensions → figure out how tall the rendered earring is
        ear_sz    = max(int(face_w * 0.22), 16)
        img_h, img_w = img.shape[:2]
        ear_drop  = int(ear_sz * img_h / max(img_w, 1))

        # Small outward nudge so earring clears the cheek
        push_px   = face_w * 0.05
        nose_x    = float(p_nose[0])

        # ── LEFT EARRING ─────────────────────────────────────────────────────────
        l_lobe_x  = float(p_l_lobe[0])
        l_lobe_y  = float(p_l_lobe[1])

        # Push left earring further left (it's left of nose)
        l_push    = -push_px if l_lobe_x <= nose_x else push_px
        l_cx_raw  = l_lobe_x + l_push
        # Center of earring image = lobe_y + half earring height → hangs downward
        l_cy_raw  = l_lobe_y + ear_drop // 2 + ear_v_offset

        l_a_raw   = -calculate_angle(p_l_up[:2], p_l_lobe[:2]) + 90.0

        sm_l  = self.ear_l_sm.smooth(cx=l_cx_raw, cy=l_cy_raw,
                                      angle=l_a_raw, scale=float(ear_sz))
        ear_l = resize_to_width(img, max(int(sm_l['scale']), 8))
        ear_l = rotate_image(ear_l, float(sm_l['angle']))
        frame = alpha_blend(frame, ear_l, float(sm_l['cx']), float(sm_l['cy']))

        # ── RIGHT EARRING (mirrored) ──────────────────────────────────────────────
        r_lobe_x  = float(p_r_lobe[0])
        r_lobe_y  = float(p_r_lobe[1])

        r_push    = push_px if r_lobe_x >= nose_x else -push_px
        r_cx_raw  = r_lobe_x + r_push
        r_cy_raw  = r_lobe_y + ear_drop // 2 + ear_v_offset

        r_a_raw   = -calculate_angle(p_r_up[:2], p_r_lobe[:2]) + 90.0

        sm_r  = self.ear_r_sm.smooth(cx=r_cx_raw, cy=r_cy_raw,
                                      angle=r_a_raw, scale=float(ear_sz))
        ear_r = resize_to_width(img, max(int(sm_r['scale']), 8))
        ear_r = cv2.flip(ear_r, 1)
        ear_r = rotate_image(ear_r, float(sm_r['angle']))
        frame = alpha_blend(frame, ear_r, float(sm_r['cx']), float(sm_r['cy']))

        return frame
