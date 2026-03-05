"""
transform_utils.py
High-precision geometry, perspective warp, smoothing engine
Optimized for Virtual Jewelry Try-On
"""

import cv2
import numpy as np
import math


# ───────────────────────── Geometry ─────────────────────────

def calculate_distance(p1, p2):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    return float(np.linalg.norm(p2 - p1))


def calculate_angle(p1, p2):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))


def unit_vector_2d(p1, p2):
    v = np.asarray(p2, float)[:2] - np.asarray(p1, float)[:2]
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([1.0, 0.0])


# ───────────────────────── Image helpers ─────────────────────────

def trim_transparency(img):

    if img is None or img.shape[2] < 4:
        return img

    alpha = img[:, :, 3]

    coords = np.argwhere(alpha > 10)

    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return img[y0:y1, x0:x1]


def resize_to_width(img, target_width):

    img = trim_transparency(img)

    if img is None:
        return img

    target_width = max(int(target_width), 4)

    h, w = img.shape[:2]

    new_h = max(int(h * target_width / w), 4)

    return cv2.resize(
        img,
        (target_width, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )


def rotate_image(img, angle_deg):

    if abs(angle_deg) < 0.3:
        return img

    h, w = img.shape[:2]

    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy

    return cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


# ───────────────────────── Perspective quad ─────────────────────────

def build_perspective_quad(cx, cy, width, height, axis_unit):

    ux, uy = axis_unit
    nx, ny = -uy, ux

    hw = width / 2
    hh = height / 2

    tl = (cx - nx*hw - ux*hh, cy - ny*hw - uy*hh)
    tr = (cx + nx*hw - ux*hh, cy + ny*hw - uy*hh)
    br = (cx + nx*hw + ux*hh, cy + ny*hw + uy*hh)
    bl = (cx - nx*hw + ux*hh, cy - ny*hw + uy*hh)

    return np.float32([tl, tr, br, bl])


def build_wrist_perspective_quad(s_idx, s_pink, s_wrist, bw, bh, depth_z=0):

    width_axis = unit_vector_2d(s_idx, s_pink)

    knuckle_mid = (np.asarray(s_idx) + np.asarray(s_pink)) / 2

    forearm_dir = unit_vector_2d(knuckle_mid, s_wrist)

    cx = float(s_wrist[0])
    cy = float(s_wrist[1])

    hw = bw / 2
    hh = bh / 2

    foreshorten = max(0.35, min(1.0, 1 + depth_z * 3))

    hh *= foreshorten

    palm = forearm_dir * hh

    tl = (cx - width_axis[0]*hw - palm[0], cy - width_axis[1]*hw - palm[1])
    tr = (cx + width_axis[0]*hw - palm[0], cy + width_axis[1]*hw - palm[1])
    br = (cx + width_axis[0]*hw + palm[0], cy + width_axis[1]*hw + palm[1])
    bl = (cx - width_axis[0]*hw + palm[0], cy - width_axis[1]*hw + palm[1])

    return np.float32([tl, tr, br, bl])


# ───────────────────────── Warp ─────────────────────────

def perspective_warp(img, dst_quad):

    try:

        h, w = img.shape[:2]

        src = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        dst = dst_quad.astype(np.float32)

        x_min = int(max(dst[:, 0].min() - 1, 0))
        y_min = int(max(dst[:, 1].min() - 1, 0))

        x_max = int(dst[:, 0].max() + 1)
        y_max = int(dst[:, 1].max() + 1)

        out_w = max(x_max - x_min, 1)
        out_h = max(y_max - y_min, 1)

        shift = np.float32([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])

        M = cv2.getPerspectiveTransform(src, dst)

        M_shift = shift @ M

        warped = cv2.warpPerspective(
            img,
            M_shift,
            (out_w, out_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        return warped, (x_min, y_min)

    except cv2.error:

        return None

# ------------------------------------------------------------
# Hand Occlusion Mask
# ------------------------------------------------------------

HAND_HULL_INDICES = [
    0, 1, 2,
    5, 9, 13, 17,
    18, 19, 20,
    4, 3,
    8, 12, 16
]


def hand_occlusion_mask(frame_shape, slm, w, h, dilate_px=6):

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)

    pts = []

    for idx in HAND_HULL_INDICES:
        lm = slm[idx]
        pts.append([int(lm.x * w), int(lm.y * h)])

    pts = np.array(pts, dtype=np.int32)

    hull = cv2.convexHull(pts)

    cv2.fillConvexPoly(mask, hull, 255)

    if dilate_px > 0:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )

        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ------------------------------------------------------------
# Apply Occlusion
# ------------------------------------------------------------

def apply_occlusion(
    frame_with_jewelry,
    original_frame,
    hand_mask,
    finger_indices,
    slm,
    w,
    h,
    occlusion_strength=0.4,
):

    if occlusion_strength <= 0:
        return frame_with_jewelry

    finger_mask = np.zeros(frame_with_jewelry.shape[:2], dtype=np.uint8)

    pts = []

    for idx in finger_indices:
        lm = slm[idx]
        pts.append([int(lm.x * w), int(lm.y * h)])

    if len(pts) < 3:
        return frame_with_jewelry

    pts = np.array(pts, dtype=np.int32)

    hull = cv2.convexHull(pts)

    cv2.fillConvexPoly(finger_mask, hull, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    eroded = cv2.erode(finger_mask, kernel, iterations=2)

    edge_mask = cv2.subtract(finger_mask, eroded)

    occ_mask = cv2.bitwise_and(hand_mask, edge_mask)

    occ_mask = occ_mask[:, :, np.newaxis].astype(np.float32) / 255.0

    result = frame_with_jewelry.astype(np.float32)

    orig = original_frame.astype(np.float32)

    result = result * (1 - occ_mask * occlusion_strength) + orig * (
        occ_mask * occlusion_strength
    )

    return result.astype(np.uint8)
# ───────────────────────── Smoothing ─────────────────────────

class ParamSmoother:

    def __init__(self, alpha=0.72):

        self.alpha = alpha
        self.state = {}

    def smooth(self, **kwargs):

        out = {}

        for k, v in kwargs.items():

            v = np.asarray(v, float)

            if k not in self.state:
                self.state[k] = v
            else:
                self.state[k] = (
                    self.alpha * v +
                    (1 - self.alpha) * self.state[k]
                )

            out[k] = self.state[k]

        return out


class AngleSmoother:

    def __init__(self, alpha=0.72):

        self.alpha = alpha
        self.cos = None
        self.sin = None

    def smooth(self, angle):

        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))

        if self.cos is None:
            self.cos = c
            self.sin = s
        else:
            self.cos = self.alpha * c + (1 - self.alpha) * self.cos
            self.sin = self.alpha * s + (1 - self.alpha) * self.sin

        return math.degrees(math.atan2(self.sin, self.cos))


class JewelrySmoother:

    def __init__(self, alpha=0.72):

        self.pos = ParamSmoother(alpha)
        self.angle = AngleSmoother(alpha)

    def smooth(self, cx, cy, scale, angle):

        p = self.pos.smooth(cx=cx, cy=cy, scale=scale)

        return {
            "cx": float(p["cx"]),
            "cy": float(p["cy"]),
            "scale": float(p["scale"]),
            "angle": self.angle.smooth(angle),
        }