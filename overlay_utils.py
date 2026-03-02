"""
overlay_utils.py
================
Vectorised alpha compositing — zero Python pixel loops.
"""

import numpy as np


def alpha_blend(background: np.ndarray, overlay: np.ndarray,
                cx: float, cy: float) -> np.ndarray:
    """
    Composite BGRA `overlay` centred at (cx, cy) onto BGR `background`.
    Handles partial out-of-frame overlays cleanly.
    """
    return _blend(background, overlay, int(round(cx - overlay.shape[1] / 2.0)),
                  int(round(cy - overlay.shape[0] / 2.0)))


def alpha_blend_at(background: np.ndarray, overlay: np.ndarray,
                   x: float, y: float) -> np.ndarray:
    """Same as alpha_blend but (x, y) is the TOP-LEFT corner."""
    return _blend(background, overlay, int(x), int(y))


def _blend(bg: np.ndarray, ov: np.ndarray, x1: int, y1: int) -> np.ndarray:
    bg_h, bg_w = bg.shape[:2]
    ov_h, ov_w = ov.shape[:2]

    x2, y2 = x1 + ov_w, y1 + ov_h
    bx1, by1 = max(x1, 0), max(y1, 0)
    bx2, by2 = min(x2, bg_w), min(y2, bg_h)
    iw, ih   = bx2 - bx1, by2 - by1
    if iw <= 0 or ih <= 0:
        return bg

    ox1, oy1  = bx1 - x1, by1 - y1
    ov_crop   = ov[oy1:oy1 + ih, ox1:ox1 + iw]
    bg_crop   = bg[by1:by2, bx1:bx2]

    if ov_crop.ndim < 3 or ov_crop.shape[2] < 4:
        bg[by1:by2, bx1:bx2] = ov_crop[:, :, :3]
        return bg

    a   = ov_crop[:, :, 3:4].astype(np.float32) / 255.0
    ov3 = ov_crop[:, :, :3].astype(np.float32)
    bg3 = bg_crop.astype(np.float32)

    bg[by1:by2, bx1:bx2] = (a * ov3 + (1.0 - a) * bg3).astype(np.uint8)
    return bg


# Back-compat alias
overlay_transparent = alpha_blend
