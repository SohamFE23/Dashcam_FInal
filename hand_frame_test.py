from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import numpy as np
import time

print("=== HAND FRAME TEST ===")

# -------------------------
# START CAMERA
# -------------------------

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 25},
    buffer_count=4
)

picam2.configure(config)
picam2.start()

time.sleep(2)

print()
print("Camera started.")
print("PUT YOUR HAND IN FRONT OF THE CAMERA NOW")
print("Waiting 5 seconds...")
time.sleep(5)

# Capture the hand scene
frame = picam2.capture_array("main")

frame = cv2.cvtColor(
    frame,
    cv2.COLOR_RGB2BGR
)

frame = np.ascontiguousarray(
    frame,
    dtype=np.uint8
)

cv2.imwrite(
    "/home/soham/Dashcam/hand_frame.jpg",
    frame
)

print("Saved:")
print("/home/soham/Dashcam/hand_frame.jpg")

picam2.stop()

print("Camera stopped.")

# -------------------------
# LOAD SAVED IMAGE
# -------------------------

frame = cv2.imread(
    "/home/soham/Dashcam/hand_frame.jpg"
)

if frame is None:
    raise RuntimeError("Could not load hand_frame.jpg")

frame = np.ascontiguousarray(
    frame,
    dtype=np.uint8
)

print("Saved frame loaded:", frame.shape)

# -------------------------
# YOLO
# -------------------------

print("Loading YOLO...")

model = YOLO(
    "/home/soham/Dashcam/best.onnx",
    task="segment"
)

print("YOLO loaded.")

# -------------------------
# REPEAT SAME HAND IMAGE
# -------------------------

for i in range(50):

    print(f"YOLO RUN {i + 1}")

    result = model(
        frame,
        conf=0.25,
        imgsz=320,
        max_det=20,
        verbose=False,
        device="cpu"
    )[0]

    print("  YOLO OK")

print()
print("=== HAND FRAME TEST COMPLETE ===")
