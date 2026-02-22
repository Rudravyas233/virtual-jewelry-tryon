import cv2
import numpy as np

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """
    Overlays a transparent PNG image onto a background image.
    
    Args:
        background: The base image (BGR).
        overlay: The image to overlay (BGRA).
        x: Top-left x coordinate.
        y: Top-left y coordinate.
        overlay_size: Tuple (width, height) to resize the overlay image.
        
    Returns:
        The combined image.
    """
    bg_h, bg_w = background.shape[:2]
    
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)
        
    h, w = overlay.shape[:2]
    
    # Check if the overlay is within the background boundaries
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background

    # Calculate overlaps
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bg_w), min(y + h, bg_h)
    
    # Calculate dimensions of the overlap
    overlay_w = x2 - x1
    overlay_h = y2 - y1
    
    if overlay_w <= 0 or overlay_h <= 0:
        return background
    
    # Extract the overlay parts that fit on the background
    overlay_crop = overlay[y1 - y:y1 - y + overlay_h, x1 - x:x1 - x + overlay_w]
    background_crop = background[y1:y2, x1:x2]
    
    # Separate alpha channel and color channels
    if overlay_crop.shape[2] < 4:
        # If no alpha channel, just paste it (though logic expects transparent PNG)
        background[y1:y2, x1:x2] = overlay_crop[:, :, :3]
        return background

    alpha = overlay_crop[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha
    
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (alpha * overlay_crop[:, :, c] +
                                      alpha_inv * background_crop[:, :, c])
        
    return background
