import cv2
import numpy as np
import os

# ==========================================================
# INPUT / OUTPUT
# ==========================================================

CATALOG_FOLDER = r"C:\Users\ANSH\Desktop\CCC\VTO\catalog_pack\catalog_pack"
OUTPUT_FOLDER = r"C:\Users\ANSH\Desktop\CCC\VTO\catalog_pack\processed_images"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

supported_ext = (".png", ".jpg", ".jpeg", ".webp")

# ==========================================================
# PROCESS LOOP
# ==========================================================

for file in os.listdir(CATALOG_FOLDER):

    if not file.lower().endswith(supported_ext):
        continue

    print(f"Processing: {file}")

    ring_path = os.path.join(CATALOG_FOLDER, file)

    ring = cv2.imread(ring_path, cv2.IMREAD_UNCHANGED)

    if ring is None:
        print("Skipping (could not load)")
        continue

    # If no alpha channel → add one
    if ring.shape[2] == 3:
        ring = cv2.cvtColor(ring, cv2.COLOR_BGR2BGRA)

    ring_rgba = cv2.cvtColor(ring, cv2.COLOR_BGRA2RGBA)

    # ======================================================
    # 1️⃣ Tight Crop (remove transparent padding)
    # ======================================================

    alpha = ring_rgba[:,:,3]
    ys, xs = np.where(alpha > 10)

    if len(xs) == 0 or len(ys) == 0:
        print("Skipping (no visible content)")
        continue

    ring_rgba = ring_rgba[ys.min():ys.max(), xs.min():xs.max()]

    # ======================================================
    # 2️⃣ Extract Front Band (same logic as your file)
    # ======================================================

    h, w = ring_rgba.shape[:2]

    top_cut = int(h * 0.30)
    bottom_cut = int(h * 0.85)

    ring_band = ring_rgba[top_cut:bottom_cut, :]

    # ======================================================
    # 3️⃣ 2x Upscale (Ultra Clean)
    # ======================================================

    ring_band = cv2.resize(
        ring_band,
        (w * 2, int((bottom_cut - top_cut) * 2)),
        interpolation=cv2.INTER_CUBIC
    )

    # ======================================================
    # 4️⃣ Save
    # ======================================================

    output_name = os.path.splitext(file)[0] + "_processed.png"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    cv2.imwrite(output_path, cv2.cvtColor(ring_band, cv2.COLOR_RGBA2BGRA))

    print(f"Saved: {output_name}")

print("\nALL RINGS PROCESSED SUCCESSFULLY ✅")
