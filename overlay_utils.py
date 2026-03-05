"""
overlay_utils.py
================
Vectorised alpha compositing — optimized for VTO rendering
"""

import numpy as np
import cv2


# --------------------------------------------------------
# High quality resize helper
# --------------------------------------------------------

def resize_overlay(overlay, width=None, height=None):
    """
    Resize overlay using high quality interpolation.
    """

    if width is None and height is None:
        return overlay

    h, w = overlay.shape[:2]

    if width is None:
        scale = height / h
        width = int(w * scale)

    if height is None:
        scale = width / w
        height = int(h * scale)

    return cv2.resize(
        overlay,
        (width, height),
        interpolation=cv2.INTER_LANCZOS4
    )


# --------------------------------------------------------
# Alpha blend (centered)
# --------------------------------------------------------

def alpha_blend(background: np.ndarray, overlay: np.ndarray,
                cx: float, cy: float) -> np.ndarray:
    """
    Composite BGRA `overlay` centred at (cx, cy) onto BGR `background`.
    """
    return _blend(
        background,
        overlay,
        int(round(cx - overlay.shape[1] / 2.0)),
        int(round(cy - overlay.shape[0] / 2.0))
    )


# --------------------------------------------------------
# Alpha blend at top-left
# --------------------------------------------------------

def alpha_blend_at(background: np.ndarray, overlay: np.ndarray,
                   x: float, y: float) -> np.ndarray:
    """
    Composite overlay where (x,y) is top-left
    """
    return _blend(background, overlay, int(x), int(y))


# --------------------------------------------------------
# Core blend function (vectorized)
# --------------------------------------------------------

def _blend(bg: np.ndarray, ov: np.ndarray, x1: int, y1: int) -> np.ndarray:

    bg_h, bg_w = bg.shape[:2]
    ov_h, ov_w = ov.shape[:2]

    x2, y2 = x1 + ov_w, y1 + ov_h

    bx1, by1 = max(x1, 0), max(y1, 0)
    bx2, by2 = min(x2, bg_w), min(y2, bg_h)

    iw, ih = bx2 - bx1, by2 - by1

    if iw <= 0 or ih <= 0:
        return bg

    ox1, oy1 = bx1 - x1, by1 - y1

    ov_crop = ov[oy1:oy1 + ih, ox1:ox1 + iw]
    bg_crop = bg[by1:by2, bx1:bx2]

    if ov_crop.ndim < 3 or ov_crop.shape[2] < 4:
        bg[by1:by2, bx1:bx2] = ov_crop[:, :, :3]
        return bg

    alpha = ov_crop[:, :, 3:4].astype(np.float32) / 255.0

    ov3 = ov_crop[:, :, :3].astype(np.float32)
    bg3 = bg_crop.astype(np.float32)

    blended = alpha * ov3 + (1.0 - alpha) * bg3

    bg[by1:by2, bx1:bx2] = blended.astype(np.uint8)

    return bg


# --------------------------------------------------------
# Perspective overlay (for rotated rings)
# --------------------------------------------------------

def perspective_overlay(frame, overlay, dst_points):
    """
    Apply perspective transform for realistic jewelry placement.
    """

    h, w = overlay.shape[:2]

    src_pts = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    dst_pts = np.float32(dst_points)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(
        overlay,
        matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LANCZOS4
    )

    return alpha_blend_at(frame, warped, 0, 0)


# --------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------

overlay_transparent = alpha_blend