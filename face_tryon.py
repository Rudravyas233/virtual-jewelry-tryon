"""
face_tryon.py
Improved Production Necklace & Earring Try-On
"""

import cv2
import mediapipe as mp
import numpy as np

from overlay_utils import alpha_blend
from transform_utils import (
    calculate_distance,
    calculate_angle,
    resize_to_width,
    rotate_image,
    ParamSmoother
)


# --------------------------------------------------
# Motion Stabilizer
# --------------------------------------------------

class MotionStabilizer:

    def __init__(self):
        self.prev = None

    def stabilize(self, value, slow=0.85, fast=0.35):

        if self.prev is None:
            self.prev = value
            return value

        delta = abs(value - self.prev)

        alpha = fast if delta > 15 else slow

        smoothed = alpha * self.prev + (1 - alpha) * value

        self.prev = smoothed
        return smoothed


# --------------------------------------------------
# Main Class
# --------------------------------------------------

class FaceTryOn:

    NOSE_TIP = 1
    CHIN = 152

    L_JAW = 234
    R_JAW = 454

    L_LOBE = 177
    R_LOBE = 401

    L_EAR_UP = 127
    R_EAR_UP = 356


    def __init__(self):

        mp_fm = mp.solutions.face_mesh

        self.face_mesh = mp_fm.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.necklace_sm = ParamSmoother(0.75)
        self.ear_l_sm = ParamSmoother(0.75)
        self.ear_r_sm = ParamSmoother(0.75)

        self.cx_stab = MotionStabilizer()
        self.cy_stab = MotionStabilizer()
        self.scale_stab = MotionStabilizer()


    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def process(self, frame, jewelry_img, jewelry_type,
                v_offset=40,
                ear_v_offset=0):

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return frame

        lm = res.multi_face_landmarks[0].landmark

        if jewelry_type == "necklace":
            return self.apply_necklace(frame, lm, jewelry_img, w, h, v_offset)

        if jewelry_type == "earrings":
            return self.apply_earrings(frame, lm, jewelry_img, w, h, ear_v_offset)

        return frame


    # --------------------------------------------------
    # Pixel helper
    # --------------------------------------------------

    def px(self, lm, idx, w, h):

        l = lm[idx]

        return np.array([l.x * w, l.y * h, l.z])


    # --------------------------------------------------
    # Necklace
    # --------------------------------------------------

    def apply_necklace(self, frame, lm, img, w, h, v_offset):

        p_ljaw = self.px(lm, self.L_JAW, w, h)
        p_rjaw = self.px(lm, self.R_JAW, w, h)
        p_chin = self.px(lm, self.CHIN, w, h)
        p_nose = self.px(lm, self.NOSE_TIP, w, h)

        face_h = abs(p_chin[1] - p_nose[1])

        jaw_dist = calculate_distance(p_ljaw[:2], p_rjaw[:2])

        cx_raw = (p_ljaw[0] + p_rjaw[0]) / 2

        cy_raw = p_chin[1] + face_h * (1.10 + v_offset / 200)

        z_avg = (p_ljaw[2] + p_rjaw[2]) / 2

        z_factor = max(0.7, min(1.3, 1.0 - z_avg * 1.5))

        scale_raw = jaw_dist * 1.20 * z_factor

        angle_raw = -calculate_angle(p_ljaw[:2], p_rjaw[:2])

        sm = self.necklace_sm.smooth(
            cx=cx_raw,
            cy=cy_raw,
            scale=scale_raw,
            angle=angle_raw
        )

        cx = self.cx_stab.stabilize(sm["cx"])
        cy = self.cy_stab.stabilize(sm["cy"])
        scale = self.scale_stab.stabilize(sm["scale"])

        nw = max(int(scale), 10)

        neck = resize_to_width(img, nw)

        neck = rotate_image(neck, float(sm["angle"]))

        return alpha_blend(frame, neck, cx, cy)


    # --------------------------------------------------
    # Earrings
    # --------------------------------------------------

    def apply_earrings(self, frame, lm, img, w, h, ear_v_offset):

        p_nose = self.px(lm, self.NOSE_TIP, w, h)

        p_l_lobe = self.px(lm, self.L_LOBE, w, h)
        p_r_lobe = self.px(lm, self.R_LOBE, w, h)

        p_l_jaw = self.px(lm, self.L_JAW, w, h)
        p_r_jaw = self.px(lm, self.R_JAW, w, h)

        p_l_up = self.px(lm, self.L_EAR_UP, w, h)
        p_r_up = self.px(lm, self.R_EAR_UP, w, h)

        face_w = calculate_distance(p_l_jaw[:2], p_r_jaw[:2])

        ear_sz = max(int(face_w * 0.22), 16)

        img_h, img_w = img.shape[:2]

        ear_drop = int(ear_sz * img_h / max(img_w, 1))

        push_px = face_w * 0.05

        nose_x = p_nose[0]

        # LEFT

        l_lobe_x = p_l_lobe[0]
        l_lobe_y = p_l_lobe[1]

        l_push = -push_px if l_lobe_x <= nose_x else push_px

        l_cx_raw = l_lobe_x + l_push
        l_cy_raw = l_lobe_y + ear_drop / 2 + ear_v_offset

        l_a_raw = -calculate_angle(p_l_up[:2], p_l_lobe[:2]) + 90

        sm_l = self.ear_l_sm.smooth(
            cx=l_cx_raw,
            cy=l_cy_raw,
            scale=ear_sz,
            angle=l_a_raw
        )

        ear_l = resize_to_width(img, int(sm_l["scale"]))

        ear_l = rotate_image(ear_l, float(sm_l["angle"]))

        frame = alpha_blend(frame, ear_l, sm_l["cx"], sm_l["cy"])


        # RIGHT

        r_lobe_x = p_r_lobe[0]
        r_lobe_y = p_r_lobe[1]

        r_push = push_px if r_lobe_x >= nose_x else -push_px

        r_cx_raw = r_lobe_x + r_push
        r_cy_raw = r_lobe_y + ear_drop / 2 + ear_v_offset

        r_a_raw = -calculate_angle(p_r_up[:2], p_r_lobe[:2]) + 90

        sm_r = self.ear_r_sm.smooth(
            cx=r_cx_raw,
            cy=r_cy_raw,
            scale=ear_sz,
            angle=r_a_raw
        )

        ear_r = resize_to_width(img, int(sm_r["scale"]))

        ear_r = cv2.flip(ear_r, 1)

        ear_r = rotate_image(ear_r, float(sm_r["angle"]))

        frame = alpha_blend(frame, ear_r, sm_r["cx"], sm_r["cy"])

        return frame