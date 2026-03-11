"""
collect_training_data.py
════════════════════════════════════════════════════════════
Collect labelled training images from your RealSense camera.

HOW TO USE:
  1. Run: python3 collect_training_data.py
  2. Hold a printed block letter in front of the camera
  3. Press the LETTER KEY on your keyboard (e.g. press 'B' for letter B)
  4. It auto-saves a burst of images to data/raw/<LETTER>/
  5. Press Q to quit

TIPS:
  - Vary distance slightly (30–70cm)
  - Rotate the card ±20 degrees
  - Try different lighting
  - Aim for 200–500 images per letter
  - Run multiple sessions across different lighting conditions

OUTPUT:
  data/raw/
    A/  → 001.png, 002.png ...
    B/  → 001.png, 002.png ...
    ...
    9/  → 001.png, 002.png ...
"""

import cv2
import numpy as np
import os
import time

SAVE_ROOT      = "data/raw"
BURST_SIZE     = 10      # Images saved per keypress
BURST_DELAY    = 0.05    # Seconds between burst frames
IMG_SIZE       = 64      # Saved image size

LABEL_MAP = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# ── Try RealSense, fall back to webcam ─────────────────────
try:
    import pyrealsense2 as rs
    USE_REALSENSE = True
except ImportError:
    USE_REALSENSE = False
    print("[INFO] pyrealsense2 not found — using webcam instead.")

def preprocess(frame):
    """Crop to centre square, greyscale, resize to 64×64."""
    h, w = frame.shape[:2]
    side  = min(h, w)
    y0    = (h - side) // 2
    x0    = (w - side) // 2
    crop  = frame[y0:y0+side, x0:x0+side]
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

def count_existing(label):
    folder = os.path.join(SAVE_ROOT, label)
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith('.png')])

def save_burst(label, frames):
    folder = os.path.join(SAVE_ROOT, label)
    os.makedirs(folder, exist_ok=True)
    existing = count_existing(label)
    for i, img in enumerate(frames):
        path = os.path.join(folder, f"{existing + i + 1:04d}.png")
        cv2.imwrite(path, img)
    print(f"  ✓ Saved {len(frames)} images for '{label}' "
          f"(total: {existing + len(frames)})")

def run():
    # ── Camera init ────────────────────────────────────────
    if USE_REALSENSE:
        pipeline = rs.pipeline()
        cfg      = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(cfg)
        def get_frame():
            frames = pipeline.wait_for_frames()
            return np.asanyarray(frames.get_color_frame().get_data())
    else:
        cap = cv2.VideoCapture(0)
        def get_frame():
            _, frame = cap.read()
            return frame

    print("\n══════════════════════════════════════════════")
    print("  Block Letter Data Collector")
    print("══════════════════════════════════════════════")
    print(f"  Press a LETTER or DIGIT key to capture {BURST_SIZE} images")
    print("  Press Escape to quit\n")

    current_label    = None
    flash_until      = 0
    counts           = {l: count_existing(l) for l in LABEL_MAP}

    try:
        while True:
            frame   = get_frame()
            display = frame.copy()
            now     = time.time()

            # Flash green on capture
            if now < flash_until:
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]),
                              (0, 255, 0), -1)
                display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

            # HUD
            cv2.putText(display,
                        f"Press a key to capture. Last: {current_label or 'none'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # Show counts
            y = 60
            for i, label in enumerate(LABEL_MAP):
                col = (0, 255, 0) if counts[label] >= 200 else (0, 180, 255)
                cv2.putText(display, f"{label}:{counts[label]}",
                            (10 + (i % 9) * 70, y + (i // 9) * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            cv2.imshow("Data Collector", display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                print("[Closing] Data collection ended by user.")
                break

            ch = chr(key).upper()
            if ch in LABEL_MAP:
                current_label = ch
                burst = []
                for _ in range(BURST_SIZE):
                    f = get_frame()
                    burst.append(preprocess(f))
                    time.sleep(BURST_DELAY)
                save_burst(current_label, burst)
                counts[current_label] = count_existing(current_label)
                flash_until = time.time() + 0.3

    finally:
        if USE_REALSENSE:
            pipeline.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()
        print("\n[Done] Total images per label:")
        for label in LABEL_MAP:
            c = count_existing(label)
            print(f"  {label}: {c} images {'✓' if c >= 200 else '⚠ need more'}")

if __name__ == '__main__':
    run()