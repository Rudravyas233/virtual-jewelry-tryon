"""
transform_utils.py
==================
Core geometry, image transformation, and smoothing engine.

World landmark strategy:
  - Use multi_hand_WORLD_landmarks for 3D geometry (angle, scale, depth)
  - Use multi_hand_landmarks (screen) for pixel position only
  This gives both metric-accurate orientation AND correct screen placement.

Upgrades vs previous version:
  - AngleSmoother: angle-wrap-safe EMA (prevents 359->1 flip jitter)
  - build_perspective_quad_3d: uses full 3-D wrist plane for realistic warp
  - hand_occlusion_mask: generate a skin-mask from convex hull of landmarks
    so jewelry rendered *behind* the hand gets correctly occluded
"""

import cv2
import numpy as np
import math


# ─────────────────────────── Geometry ─────────────────────────────────────────

def calculate_distance(p1, p2):
    """Euclidean distance between two points (2D or 3D numpy arrays)."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    return float(np.linalg.norm(p2 - p1))


def calculate_angle(p1, p2):
    """
    Angle (degrees) of the 2-D vector from p1 → p2.
    Screen convention: Y grows downward.
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))


def unit_vector_2d(p1, p2):
    """Unit vector from p1 → p2 (2-D)."""
    v = np.asarray(p2, float)[:2] - np.asarray(p1, float)[:2]
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([1.0, 0.0])


def project_width(center, p_left, p_right, axis_unit):
    """
    Return the span of two flanking points projected onto the perpendicular
    of `axis_unit`.  Used to estimate true cross-section widths.
    """
    perp   = np.array([-axis_unit[1], axis_unit[0]])
    c      = np.asarray(center, float)[:2]
    proj_l = float(np.dot(np.asarray(p_left,  float)[:2] - c, perp))
    proj_r = float(np.dot(np.asarray(p_right, float)[:2] - c, perp))
    return abs(proj_l - proj_r)


# ─────────────────────────── Image helpers ────────────────────────────────────

def trim_transparency(img):
    """Crop BGRA image tightly to the bounding box of visible pixels."""
    if img is None or img.ndim < 3 or img.shape[2] < 4:
        return img
    alpha  = img[:, :, 3]
    coords = np.argwhere(alpha > 10)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def resize_to_width(img, target_width):
    """Resize BGRA image to target_width, maintaining aspect ratio."""
    img = trim_transparency(img)
    if img is None or img.shape[1] == 0:
        return img
    target_width = max(int(target_width), 4)
    h, w         = img.shape[:2]
    target_h     = max(int(h * target_width / w), 4)
    return cv2.resize(img, (target_width, target_h), interpolation=cv2.INTER_AREA)


def rotate_image(img, angle_deg):
    """
    Rotate BGRA image by angle_deg around its centre.
    Canvas is expanded so no content is clipped.
    """
    if abs(angle_deg) < 0.3:
        return img
    h, w  = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M      = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a  = abs(M[0, 0])
    sin_a  = abs(M[0, 1])
    new_w  = int(h * sin_a + w * cos_a)
    new_h  = int(h * cos_a + w * sin_a)
    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))


def build_perspective_quad(cx, cy, width, height, axis_unit):
    """
    Build 4 destination corners (TL, TR, BR, BL) for a rectangle of
    (width × height) centred at (cx, cy) and oriented along axis_unit.

    axis_unit : 2-D unit vector along the "length" axis of the jewelry.
    """
    ux, uy = axis_unit[0], axis_unit[1]
    nx, ny = -uy, ux                  # perpendicular (across)
    hw, hh = width / 2.0, height / 2.0

    tl = (cx - nx*hw - ux*hh, cy - ny*hw - uy*hh)
    tr = (cx + nx*hw - ux*hh, cy + ny*hw - uy*hh)
    br = (cx + nx*hw + ux*hh, cy + ny*hw + uy*hh)
    bl = (cx - nx*hw + ux*hh, cy - ny*hw + uy*hh)
    return np.float32([tl, tr, br, bl])


def build_wrist_perspective_quad(s_idx, s_pink, s_wrist, bw, bh, depth_z=0.0):
    """
    Build a perspective quad for bracelet using three landmark screen points:
      s_idx   — index MCP (screen px)
      s_pink  — pinky MCP (screen px)
      s_wrist — wrist center (screen px)

    Geometry:
      • width_axis  = unit vector from s_idx → s_pink  (across the wrist)
      • forearm_dir = unit vector from knuckle-midpoint → s_wrist
                      (pointing toward forearm, i.e. downward when hand is up)

    The bracelet quad is symmetric: equal hh on both sides of the centre.
    Foreshortening (depth_z) compresses the height when the wrist tilts.

    bw, bh  — target bracelet pixel width and height
    depth_z — wrist.z from world landmarks (negative = closer to camera)
    """
    # ── Width axis: across the wrist (index knuckle → pinky knuckle) ──────────
    width_axis = unit_vector_2d(s_idx, s_pink)

    # ── Forearm direction: from knuckle midpoint TOWARD wrist ─────────────────
    knuckle_mid = (np.asarray(s_idx, float) + np.asarray(s_pink, float)) / 2.0
    forearm_dir = unit_vector_2d(knuckle_mid, s_wrist)  # points toward forearm

    cx = float(s_wrist[0])
    cy = float(s_wrist[1])
    hw = bw / 2.0
    hh = bh / 2.0

    # ── Depth-based foreshortening ─────────────────────────────────────────────
    # depth_z < 0 → wrist toward camera → appears more foreshortened on height
    # Clamp to sensible range
    foreshortenH = max(0.35, min(1.0, 1.0 + depth_z * 3.0))
    hh_scaled = hh * foreshortenH

    # ── Build symmetric quad centred at wrist ─────────────────────────────────
    # The bracelet straddles the wrist: half toward palm, half toward forearm
    # palm_side  = -forearm_dir (pointing toward palm / knuckles)
    # forearm_side =  forearm_dir (pointing toward arm)
    palm_off    = forearm_dir * hh_scaled   # offset from centre toward forearm

    tl = (cx - width_axis[0]*hw - palm_off[0],
          cy - width_axis[1]*hw - palm_off[1])
    tr = (cx + width_axis[0]*hw - palm_off[0],
          cy + width_axis[1]*hw - palm_off[1])
    br = (cx + width_axis[0]*hw + palm_off[0],
          cy + width_axis[1]*hw + palm_off[1])
    bl = (cx - width_axis[0]*hw + palm_off[0],
          cy - width_axis[1]*hw + palm_off[1])

    return np.float32([tl, tr, br, bl])


def perspective_warp(img, dst_quad):
    """
    Warp `img` (BGRA) so its four corners map to `dst_quad` (4×2 float32).
    Returns (warped_image, (x_origin, y_origin)) or None on failure.
    """
    try:
        h, w   = img.shape[:2]
        src    = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst    = dst_quad.astype(np.float32)
        x_min  = int(max(dst[:, 0].min() - 1, 0))
        y_min  = int(max(dst[:, 1].min() - 1, 0))
        x_max  = int(dst[:, 0].max() + 1)
        y_max  = int(dst[:, 1].max() + 1)
        out_w  = max(x_max - x_min, 1)
        out_h  = max(y_max - y_min, 1)

        shift  = np.float32([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        M      = cv2.getPerspectiveTransform(src, dst)
        M_s    = shift @ M

        warped = cv2.warpPerspective(img, M_s, (out_w, out_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0, 0))
        return warped, (x_min, y_min)
    except cv2.error:
        return None


# ─────────────────────────── Occlusion Mask ───────────────────────────────────

# Indices of hand landmarks that form the outer boundary of the hand
# (convex hull seed for occlusion masking)
HAND_HULL_INDICES = [
    0,   # wrist
    1,   # thumb CMC
    2,   # thumb MCP
    5,   # index MCP
    9,   # middle MCP
    13,  # ring MCP
    17,  # pinky MCP
    18,  # pinky PIP
    19,  # pinky DIP
    20,  # pinky tip
    4,   # thumb tip
    3,   # thumb DIP
    8,   # index tip
    12,  # middle tip
    16,  # ring tip
]


def hand_occlusion_mask(frame_shape, slm, w, h, dilate_px: int = 6):
    """
    Build a binary mask (uint8, same HxW as frame) covering the hand.

    The mask is 255 on hand-skin pixels and 0 elsewhere.
    It is used to re-apply hand pixels ON TOP of the blended jewelry,
    producing an occlusion effect where jewelry appears behind the hand.

    dilate_px : expand the mask outward to account for skin tone variation
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts  = []
    for idx in HAND_HULL_INDICES:
        lm = slm[idx]
        pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def apply_occlusion(frame_with_jewelry: np.ndarray,
                    original_frame: np.ndarray,
                    hand_mask: np.ndarray,
                    finger_indices,
                    slm, w, h,
                    occlusion_strength: float = 0.40) -> np.ndarray:
    """
    Lightly re-composite hand skin OVER jewelry at the finger/wrist edges.

    occlusion_strength controls how strongly skin is painted back:
      0.0 = no occlusion (jewelry fully on top)
      1.0 = full occlusion (skin completely covers jewelry edges)
      0.4 = default — subtle wrap-behind effect without hiding the ring

    finger_indices: landmark indices defining the narrow edge region where
                    the jewelry should appear to go 'behind' the skin.
    """
    if occlusion_strength <= 0:
        return frame_with_jewelry

    # Build a ring-of-pixels edge mask — ERODE the full finger hull then
    # subtract it, leaving only the boundary strip where skin meets jewelry.
    finger_mask = np.zeros(frame_with_jewelry.shape[:2], dtype=np.uint8)
    pts = []
    for idx in finger_indices:
        lm = slm[idx]
        pts.append([int(lm.x * w), int(lm.y * h)])
    if len(pts) < 3:
        return frame_with_jewelry

    pts  = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(finger_mask, hull, 255)

    # Erode to get inner region, then subtract → thin boundary ring only
    k_big    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    eroded   = cv2.erode(finger_mask, k_big, iterations=2)
    edge_mask = cv2.subtract(finger_mask, eroded)       # only outer edge strip

    occ_mask = cv2.bitwise_and(hand_mask, edge_mask)
    occ_3ch  = occ_mask[:, :, np.newaxis].astype(np.float32) / 255.0 * occlusion_strength

    result = frame_with_jewelry.astype(np.float32)
    orig   = original_frame.astype(np.float32)
    result = result * (1.0 - occ_3ch) + orig * occ_3ch
    return result.astype(np.uint8)


# ─────────────────────────── Temporal Smoothing ───────────────────────────────

class ParamSmoother:
    """
    Exponential Moving Average over a named dict of scalar / array parameters.
    alpha ∈ [0.6, 0.8]  — higher = faster response, lower = more stable.

    Angles are treated as ordinary scalars here; use AngleSmoother for
    angle-wrap-safe smoothing of rotation parameters.
    """
    def __init__(self, alpha: float = 0.70):
        self.alpha  = float(alpha)
        self._state = {}

    def smooth(self, **kwargs) -> dict:
        out = {}
        for k, v in kwargs.items():
            v = np.asarray(v, dtype=float)
            if k not in self._state:
                self._state[k] = v.copy()
            else:
                self._state[k] = self.alpha * v + (1.0 - self.alpha) * self._state[k]
            out[k] = self._state[k].copy()
        return out

    def reset(self):
        self._state.clear()


class AngleSmoother:
    """
    Angle-wrap-safe EMA for rotation angles (in degrees).

    Prevents the 359° → 1° jump that causes jitter when a ring or bracelet
    crosses the 0/360 boundary.  Uses the circular mean technique:
    average the unit vectors cos(θ), sin(θ) and convert back.
    """
    def __init__(self, alpha: float = 0.70):
        self.alpha = float(alpha)
        self._cos  = None
        self._sin  = None

    def smooth(self, angle_deg: float) -> float:
        c = math.cos(math.radians(angle_deg))
        s = math.sin(math.radians(angle_deg))
        if self._cos is None:
            self._cos, self._sin = c, s
        else:
            self._cos = self.alpha * c + (1.0 - self.alpha) * self._cos
            self._sin = self.alpha * s + (1.0 - self.alpha) * self._sin
        return math.degrees(math.atan2(self._sin, self._cos))

    def reset(self):
        self._cos = self._sin = None


class JewelrySmoother:
    """
    Combined smoother: EMA for (cx, cy, scale) + angle-wrap-safe for angle.
    """
    def __init__(self, alpha: float = 0.70):
        self._pos   = ParamSmoother(alpha)
        self._angle = AngleSmoother(alpha)

    def smooth(self, cx, cy, scale, angle) -> dict:
        ps = self._pos.smooth(cx=cx, cy=cy, scale=scale)
        return {
            'cx':    float(ps['cx']),
            'cy':    float(ps['cy']),
            'scale': float(ps['scale']),
            'angle': self._angle.smooth(float(angle)),
        }

    def reset(self):
        self._pos.reset()
        self._angle.reset()
