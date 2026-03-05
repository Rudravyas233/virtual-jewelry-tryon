"""
hand_tryon.py
=============
Production-Level Ring & Bracelet Virtual Try-On — 3D-Aware Pipeline.

Pipeline per frame
──────────────────
1.  Detect hand with MediaPipe Hands
2.  Extract BOTH screen landmarks (pixel pos) + world landmarks (3-D metric)
3.  Compute center, scale, rotation from world + screen landmarks
4.  Apply rotation + perspective warp (cv2.getPerspectiveTransform)
5.  Apply exponential smoothing (position, scale, angle-wrap-safe angle)
6.  Alpha-blend jewelry onto frame (vectorised, no pixel loops)
7.  Apply occlusion mask → re-paint hand skin over jewelry where needed
8.  Return final composited frame

Key design decisions
────────────────────
• RING
    - Center at 30% along MCP→PIP (anatomical proximal phalanx)
    - Width derived from 3-D world perpendicular projection (true cross-section)
    - Perspective quad built from finger axis + perpendicular
    - Occlusion mask from finger convex hull re-applied after blend

• BRACELET
    - Width = 0.75 × screen knuckle span (wrist narrower than knuckles)
    - Perspective quad uses build_wrist_perspective_quad() which takes
      all three wrist-plane points (0, 5, 17) for realistic foreshortening
    - Trapezoid warp applied first to simulate cylindrical wrap
    - Depth-adjusted from wrist world z

• Smoothers
    - JewelrySmoother used everywhere (angle-wrap-safe EMA)
    - alpha = 0.68 → good balance between snap and stability
"""

import math
import cv2
import mediapipe as mp
import numpy as np

from overlay_utils import alpha_blend, alpha_blend_at
from transform_utils import (
    calculate_distance, calculate_angle,
    unit_vector_2d,
    resize_to_width, rotate_image,
    build_perspective_quad,
    build_wrist_perspective_quad,
    perspective_warp,
    hand_occlusion_mask,
    apply_occlusion,
    JewelrySmoother,
)


# ── Finger config ─────────────────────────────────────────────────────────────
# (MCP_idx, PIP_idx, DIP_idx, left_neighbour_MCP, right_neighbour_MCP,
#  width_factor,  finger_landmark_hull_for_occlusion)
FINGER_CONFIG = {
    #          MCP  PIP  DIP  L_nbr R_nbr  wf   occlusion hull indices
    "Thumb":  (2,   3,   4,   0,    5,   0.40, [1, 2, 3, 4]),
    "Index":  (5,   6,   7,   2,    9,   0.50, [5, 6, 7, 8, 9]),
    "Middle": (9,   10,  11,  5,    13,  0.50, [5, 9, 10, 11, 12, 13]),
    "Ring":   (13,  14,  15,  9,    17,  0.55, [9, 13, 14, 15, 16, 17]),
    "Pinky":  (17,  18,  19,  13,   17,  0.45, [13, 17, 18, 19, 20]),
}

# ── EMA alpha ─────────────────────────────────────────────────────────────────
_ALPHA = 0.68


def _project_width_3d(center_3d, p_left_3d, p_right_3d, axis_unit_2d):
    """Span of two 3-D flanking points on the perpendicular of axis_unit_2d."""
    perp   = np.array([-axis_unit_2d[1], axis_unit_2d[0]])
    c2     = np.asarray(center_3d, float)[:2]
    proj_l = float(np.dot(np.asarray(p_left_3d,  float)[:2] - c2, perp))
    proj_r = float(np.dot(np.asarray(p_right_3d, float)[:2] - c2, perp))
    return abs(proj_l - proj_r)


def trapezoid_warp(img, taper: float = 0.10):
    """
    Perspective-squeeze to simulate cylindrical wrapping of a bracelet.
    The top edge is compressed inward by `taper` fraction of width.

        ┌────────────────┐         ┌──────────────┐
        │  flat image    │   →     │ curved image  │
        └────────────────┘         └──────────────┘
    """
    if taper <= 0 or img is None:
        return img
    h, w = img.shape[:2]
    dx = int(w * taper)
    src = np.float32([[0, 0],    [w, 0],    [w, h],    [0, h]])
    dst = np.float32([[dx, 0],   [w-dx, 0], [w, h],    [0, h]])
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))


# ── Main class ────────────────────────────────────────────────────────────────

class HandTryOn:
    WRIST      = 0
    THUMB_CMC  = 1
    THUMB_MCP  = 2
    INDEX_MCP  = 5
    MIDDLE_MCP = 9
    RING_MCP   = 13
    PINKY_MCP  = 17

    def __init__(self):
        mp_h = mp.solutions.hands
        self.hands = mp_h.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )
        # One JewelrySmoother per finger (angle-wrap safe)
        self.ring_sm     = {f: JewelrySmoother(_ALPHA) for f in FINGER_CONFIG}
        self.bracelet_sm = JewelrySmoother(_ALPHA)

    # ── Public ────────────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray,
                jewelry_img: np.ndarray,
                jewelry_type: str,
                finger: str = "Ring",
                ring_scale: float = 1.0,
                bracelet_scale: float = 1.0) -> np.ndarray:

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self.hands.process(rgb)

        if (not res
                or not res.multi_hand_landmarks
                or not res.multi_hand_world_landmarks):
            return frame

        slm = res.multi_hand_landmarks[0].landmark        # screen (normalised)
        wlm = res.multi_hand_world_landmarks[0].landmark  # world (metres)

        if jewelry_type == 'ring':
            f = finger if finger in FINGER_CONFIG else "Ring"
            return self._apply_ring(frame, slm, wlm, jewelry_img, w, h, f, ring_scale)
        if jewelry_type == 'bracelet':
            return self._apply_bracelet(frame, slm, wlm, jewelry_img, w, h, bracelet_scale)
        return frame

    # ── Helper: screen pixel from screen landmark ─────────────────────────────
    def _spx(self, lm, idx, w, h) -> np.ndarray:
        l = lm[idx]
        return np.array([l.x * w, l.y * h])

    # ── Helper: world 3-D point from world landmark ───────────────────────────
    def _wpx(self, lm, idx) -> np.ndarray:
        l = lm[idx]
        return np.array([l.x, l.y, l.z])

    # ═══════════════════════════════════════════════════════════════════════════
    # RING
    # ═══════════════════════════════════════════════════════════════════════════
    def _apply_ring(self, frame, slm, wlm, img, w, h, finger_name, scale_mul):
        """
        3-D aware ring fitting pipeline:

        1. Extract screen landmarks for pixel positions
        2. Extract world landmarks for 3-D direction + depth
        3. Compute finger cross-section width from world perpendicular projection
        4. Convert world width → screen pixels via px/metre ratio
        5. Apply depth correction (world z)
        6. Compute rotation from screen finger axis
        7. EMA smooth (position, scale, angle-wrap-safe angle)
        8. Build perspective quad from finger axis + perpendicular
        9. Warp jewelry → quad
        10. Alpha blend
        11. Re-apply hand occlusion mask (jewelry wraps behind finger)
        """
        cfg = FINGER_CONFIG[finger_name]
        mcp_i, pip_i, dip_i, l_i, r_i, w_factor, occ_hull = cfg

        # ── Screen landmarks (pixel positions) ─────────────────────────────
        s_mcp = self._spx(slm, mcp_i, w, h)
        s_pip = self._spx(slm, pip_i, w, h)

        # ── World landmarks (3-D metric, metres) ───────────────────────────
        w_mcp = self._wpx(wlm, mcp_i)
        w_pip = self._wpx(wlm, pip_i)
        w_dip = self._wpx(wlm, dip_i)
        w_l   = self._wpx(wlm, l_i)
        w_r   = self._wpx(wlm, r_i)

        # ── Screen segment vector ───────────────────────────────────────────
        seg_vec = s_pip - s_mcp
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 2:
            return frame

        # ── Finger WIDTH from 3-D world perpendicular projection ───────────
        # The world axis (XY) gives us the direction of the finger in metric space
        world_dir_3d  = w_dip - w_mcp
        world_axis_2d = world_dir_3d[:2] / (np.linalg.norm(world_dir_3d[:2]) + 1e-9)
        cross_dist    = _project_width_3d(w_mcp, w_l, w_r, world_axis_2d)
        finger_w_world = cross_dist * w_factor   # world metres

        # pixels-per-metre from screen/world segment ratio
        world_seg_len = np.linalg.norm(w_pip - w_mcp) + 1e-9
        px_per_m      = seg_len / world_seg_len
        finger_w_px   = finger_w_world * px_per_m

        # ── Depth correction using world z ─────────────────────────────────
        # z < 0  → finger toward camera → appears larger
        # z > 0  → finger away          → appears smaller
        depth_z   = float(w_mcp[2])
        depth_fac = max(0.60, min(1.50, 1.0 - depth_z * 5.5))
        ring_w_raw = max(finger_w_px * depth_fac * scale_mul, 8.0)

        # ── Ring centre: 45% along MCP→PIP (slightly up the finger) ──────────
        t       = 0.45
        cx_raw  = float(s_mcp[0] + t * seg_vec[0])
        cy_raw  = float(s_mcp[1] + t * seg_vec[1])

        # ── Rotation angle from screen finger axis ─────────────────────────
        # atan2(dy, dx) gives the actual screen orientation of the finger
        screen_dir = unit_vector_2d(s_mcp, s_pip)
        angle_raw  = float(np.degrees(np.arctan2(screen_dir[1], screen_dir[0]))) + 90.0

        # ── EMA smooth (angle-wrap-safe) ───────────────────────────────────
        sm    = self.ring_sm[finger_name].smooth(
            cx=cx_raw, cy=cy_raw, scale=ring_w_raw, angle=angle_raw)
        cx    = float(sm['cx'])
        cy    = float(sm['cy'])
        rw    = max(int(sm['scale']), 8)
        angle = float(sm['angle'])

        # ── Perspective quad from finger axis ──────────────────────────────
        # The quad is anchored to the finger plane, making the ring
        # appear to lie flat ON the finger (not floating)
        axis  = unit_vector_2d(s_mcp, s_pip)
        rh    = int(rw * img.shape[0] / max(img.shape[1], 1))
        quad  = build_perspective_quad(cx, cy, rw, rh, axis)

        # Save original before blending (for occlusion)
        original_frame = frame.copy()

        # ── Perspective warp + alpha blend ─────────────────────────────────
        result = perspective_warp(img, quad)
        if result is not None:
            warped, origin = result
            frame = alpha_blend_at(frame, warped, origin[0], origin[1])
        else:
            # Fallback: rotate and centre-blend
            proc = resize_to_width(img, rw)
            proc = rotate_image(proc, angle)
            frame = alpha_blend(frame, proc, cx, cy)

        # ── Occlusion mask: re-apply finger skin OVER ring edges ───────────
        # This creates a subtle wrap-around at the finger boundary.
        # occlusion_strength=0.40: only 40% of edge pixels are repainted,
        # so the ring body stays clearly visible.
        try:
            hand_mask = hand_occlusion_mask(frame.shape, slm, w, h, dilate_px=4)
            frame = apply_occlusion(frame, original_frame, hand_mask,
                                    occ_hull, slm, w, h, occlusion_strength=0.40)
        except Exception:
            pass   # Never crash — occlusion is enhancement, not critical

        return frame

    # ═══════════════════════════════════════════════════════════════════════════
    # BRACELET
    # ═══════════════════════════════════════════════════════════════════════════
    def _apply_bracelet(self, frame, slm, wlm, img, w, h, scale_mul):
        """
        3-D aware bracelet fitting pipeline:

        1. Extract screen landmarks: wrist (0), index MCP (5), pinky MCP (17),
           middle MCP (9), thumb MCP (2)
        2. Extract world landmarks for depth correction
        3. Compute wrist width = 0.75 × knuckle span (index–pinky)
        4. Apply depth correction from wrist world z
        5. Compute rotation from wrist axis (perpendicular to forearm)
        6. EMA smooth (angle-wrap-safe)
        7. Apply trapezoid warp to simulate cylindrical curvature
        8. Build wrist perspective quad using 3-point wrist plane
           (uses build_wrist_perspective_quad for realistic foreshortening)
        9. Perspective warp + alpha blend
        10. Occlusion mask (wrist region)
        """
        # ── Screen landmarks ────────────────────────────────────────────────
        s_wrist = self._spx(slm, self.WRIST,      w, h)
        s_idx   = self._spx(slm, self.INDEX_MCP,  w, h)
        s_pink  = self._spx(slm, self.PINKY_MCP,  w, h)
        s_mid   = self._spx(slm, self.MIDDLE_MCP, w, h)
        s_tmcp  = self._spx(slm, self.THUMB_MCP,  w, h)

        # ── World landmarks ─────────────────────────────────────────────────
        w_wrist = self._wpx(wlm, self.WRIST)

        # ── Wrist width: 0.75 × screen knuckle span ─────────────────────────
        # Real wrist is ~75% of the width between index and pinky knuckles
        knuckle_span   = calculate_distance(s_idx, s_pink)
        depth_z        = float(w_wrist[2])
        depth_fac      = max(0.60, min(1.50, 1.0 - depth_z * 4.5))
        bracelet_w_raw = max(knuckle_span * 0.75 * depth_fac * scale_mul, 12.0)

        # ── Centre: AT the wrist landmark ──────────────────────────────────
        cx_raw = float(s_wrist[0])
        cy_raw = float(s_wrist[1])

        # ── Rotation: ALONG the wrist width axis (index→pinky direction) ────────
        # The bracelet is a band that lies ACROSS the wrist.
        # Its texture runs left-right (along wrist width), so we rotate by the
        # angle of the index→pinky vector, NOT the forearm axis.
        wrist_width_vec = unit_vector_2d(s_idx, s_pink)
        angle_raw = float(np.degrees(np.arctan2(wrist_width_vec[1], wrist_width_vec[0])))

        # ── EMA smooth (angle-wrap-safe) ───────────────────────────────────
        sm    = self.bracelet_sm.smooth(
            cx=cx_raw, cy=cy_raw, scale=bracelet_w_raw, angle=angle_raw)
        cx    = float(sm['cx'])
        cy    = float(sm['cy'])
        bw    = max(int(sm['scale']), 12)
        angle = float(sm['angle'])

        # ── Trapezoid warp: cylinder curvature simulation ───────────────────
        # Taper amount depends on how horizontal the wrist width axis is.
        # When wrist is horizontal → strong taper; vertical → less taper.
        wrist_axis_angle = abs(
            math.degrees(math.atan2(
                float(wrist_width_vec[1]),
                float(wrist_width_vec[0])
            ))
        ) % 90
        # taper 0.03 (vertical wrist) → 0.08 (horizontal wrist)
        taper    = 0.03 + 0.05 * (wrist_axis_angle / 90.0)
        proc_img = resize_to_width(img, bw)
        proc_img = trapezoid_warp(proc_img, taper=taper)

        # ── Wrist-plane perspective quad ────────────────────────────────────
        # Uses index MCP, pinky MCP, and wrist to define the wrist plane,
        # producing realistic foreshortening when wrist tilts toward camera
        # ── Wrist anchor: shift down the forearm ────────────────────────────
        # Only move s_wrist — s_idx and s_pink stay untouched so width/angle
        # stay correct. This simply lowers the bracelet center.
        knuckle_mid  = (s_idx + s_pink) / 2.0
        fw_vec       = s_wrist - knuckle_mid          # knuckles → wrist
        fw_unit      = fw_vec / (np.linalg.norm(fw_vec) + 1e-9)
        shift_px     = np.linalg.norm(fw_vec) * 0.30  # 30% of knuckle→wrist length
        s_wrist_low  = s_wrist + fw_unit * shift_px   # push further toward forearm

        bh   = int(bw * proc_img.shape[0] / max(proc_img.shape[1], 1))
        quad = build_wrist_perspective_quad(s_idx, s_pink, s_wrist_low,
                                            bw, bh, depth_z=depth_z)

        # Save original frame for occlusion
        original_frame = frame.copy()

        # ── Perspective warp + alpha blend ─────────────────────────────────
        result = perspective_warp(proc_img, quad)
        if result is not None:
            warped, origin = result
            frame = alpha_blend_at(frame, warped, origin[0], origin[1])
        else:
            # Fallback: rotate and centre-blend
            rotated = rotate_image(proc_img, angle)
            frame   = alpha_blend(frame, rotated, cx, cy)

        # ── Occlusion mask: wrist region ───────────────────────────────────
        # Re-apply a light skin tone at bracelet edges for natural wrap-around
        try:
            occlusion_hull = [
                self.WRIST, self.THUMB_CMC, self.THUMB_MCP,
                self.INDEX_MCP, self.PINKY_MCP
            ]
            hand_mask = hand_occlusion_mask(frame.shape, slm, w, h, dilate_px=5)
            frame = apply_occlusion(frame, original_frame, hand_mask,
                                    occlusion_hull, slm, w, h, occlusion_strength=0.35)
        except Exception:
            pass

        return frame
