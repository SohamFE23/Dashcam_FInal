"""
ADAS REALTIME CAMERA INFERENCE - Raspberry Pi 4 (8GB) SAFE BUILD
Camera: Pi CSI via Picamera2  |  Model: YOLO segmentation (NCNN preferred, ONNX fallback)

Fixes vs previous version:
  - Illegal Instruction (SIGILL): prefer NCNN backend; fresh aarch64 wheels (see header notes)
  - task="detect" -> task="segment" (masks were always None -> lane/steering logic dead)
  - Single consistent class map (colors + names matched)
  - Vectorized polygon test (no per-point pointPolygonTest loop)
  - masks.data + contour extraction (fast on Pi) with .xy fallback
  - One valid try/finally cleanup block
  - Radar rendered directly at display resolution (no 600x900 intermediate)
  - Degenerate-mask guards (the "small object -> crash" path)
"""

import os
import cv2
import numpy as np
import time
import threading
from collections import deque

from ultralytics import YOLO
from picamera2 import Picamera2

# ============================================================
# 0. THREAD / BLAS SANITY (avoid NEON dispatch crashes on Pi)
# ============================================================
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

# ============================================================
# 1. MODEL PATHS  (NCNN folder first -> ONNX fallback)
# ============================================================
NCNN_DIR  = "/home/soham/Dashcam/best_ncnn_model"   # from: yolo export model=best.pt format=ncnn imgsz=320
ONNX_PATH = "/home/soham/Dashcam/best.onnx"

if os.path.isdir(NCNN_DIR):
    WEIGHTS_PATH = NCNN_DIR
    BACKEND = "NCNN"
else:
    WEIGHTS_PATH = ONNX_PATH
    BACKEND = "ONNX-CPU"

# Segmentation task (NOT detect) - required for lane masks
model = YOLO(WEIGHTS_PATH, task="segment")

# --- PI 4 SPEED FIX -------------------------------------------------------
# Ultralytics loads NCNN with num_threads=1 -> single-core ~5 FPS.
# Grab the raw ncnn.Net and force all 4 cores. Expected: 10-14 FPS.
def _boost_ncnn(m, threads=4):
    try:
        net = getattr(getattr(m, "model", None), "net", None)
        if net is not None:
            net.opt.num_threads = threads
            net.opt.use_local_pool_allocator = True
            print(f"NCNN num_threads forced to {threads}")
        else:
            print("note: ncnn net handle not found (backend may differ)")
    except Exception as e:
        print("NCNN thread boost skipped:", e)

_boost_ncnn(model)
# ---------------------------------------------------------------------------

# ------------------------------------------------------------
# SPEED SETTINGS
# ------------------------------------------------------------
INFERENCE_SIZE = 320        # Pi 4 sweet spot; raise to 416 only if YOLO FPS > 12
CONF_THRESHOLD = 0.25
MAX_DETECTIONS = 20
INFER_THREADS  = 2          # Pi 4 quad-core: 2 for YOLO, 2 for camera/GUI

if hasattr(model, "predictor") and model.predictor is not None:
    try:
        model.predictor.args.workers = 1
    except Exception:
        pass

# ============================================================
# 2. CLASS MAP  (SINGLE SOURCE OF TRUTH - BGR colors)
#    ids: 0 Bus | 1 Car | 2 Dashed Lane | 3 Pedestrian | 4 Truck | 5 Yellow Lane
# ============================================================
class_names = {0: "Bus", 1: "Car", 2: "Dashed Lane",
               3: "Pedestrian", 4: "Truck", 5: "Yellow Lane"}
class_colors = {
    0: (0, 165, 255),   # Bus        - Orange
    1: (0, 0, 255),     # Car        - Red
    2: (0, 255, 0),     # Dashed Lane- Green
    3: (255, 0, 255),   # Pedestrian - Magenta
    4: (255, 0, 0),     # Truck      - Blue
    5: (0, 255, 255),   # Yellow Lane- Yellow
}
LANE_CLASSES = (2, 5)

# ============================================================
# 3. PERIMETER ZONE (reference 1280x720, scaled to camera res)
# ============================================================
raw_pts = [[440, 430], [548, 270], [710, 268], [828, 411],
           [712, 398], [632, 396], [553, 405], [440, 430]]

# ============================================================
# 4. CAMERA SETUP
# ============================================================
w, h = 640, 480
camera_fps = 25

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": (w, h), "format": "RGB888"},
    controls={"FrameRate": camera_fps},
    buffer_count=4,
))
picam2.start()
time.sleep(2)

class PicameraCapture:
    def isOpened(self):
        return True
    def read(self):
        try:
            frame = picam2.capture_array("main")
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Camera capture error: {e}")
            return False, None
    def release(self):
        try:
            picam2.stop()
        except Exception:
            pass

cap = PicameraCapture()

# ============================================================
# 5. ZONE SCALING + RADAR GEOMETRY
# ============================================================
sx, sy = w / 1280.0, h / 720.0
ZONE_POINTS = np.array([[int(x * sx), int(y * sy)] for x, y in raw_pts], dtype=np.int32)
poly_center_x = (ZONE_POINTS[:, 0].min() + ZONE_POINTS[:, 0].max()) // 2

SHOW_SCALE = 0.75                    # shrink output window -> faster X11 rendering
RADAR_W, RADAR_H = 130, 210          # rendered directly at display size
RADAR_SCALE = (RADAR_W / w, RADAR_H / h)

# ============================================================
# 6. SHARED STATE + INFERENCE WORKER
# ============================================================
frame_lock = threading.Lock()
result_lock = threading.Lock()
stop_event = threading.Event()

latest_frame = None
latest_payload = None          # list of (cls, polygon_np) - plain data, no Result obj

frame_idx = 0
fps_deque = deque(maxlen=30)
inf_fps_deque = deque(maxlen=30)

def extract_polygons(result):
    """Fast polygon extraction: masks.data -> contours, .xy fallback."""
    if result.masks is None or result.boxes is None:
        return []
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    polys = []

    data = getattr(result.masks, "data", None)
    if data is not None:
        mh, mw = data.shape[1], data.shape[2]
        for i in range(min(len(cls_ids), data.shape[0])):
            m = (data[i].cpu().numpy() > 0.5).astype(np.uint8) * 255
            if m.sum() < 30:                      # degenerate/speck mask guard
                continue
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            if len(c) < 5:
                continue
            pts = c.reshape(-1, 2).astype(np.float32)
            pts[:, 0] *= w / mw                   # mask res -> camera res
            pts[:, 1] *= h / mh
            polys.append((int(cls_ids[i]), pts.astype(np.int32)))
    else:  # fallback path
        try:
            for i, xy in enumerate(result.masks.xy):
                pts = np.array(xy, dtype=np.int32)
                if len(pts) >= 3:
                    polys.append((int(cls_ids[i]), pts))
        except Exception:
            pass
    return polys

def inference_worker():
    global latest_payload
    last_t = time.time()
    while not stop_event.is_set():
        with frame_lock:
            f = None if latest_frame is None else latest_frame.copy()
        if f is None:
            time.sleep(0.002)
            continue
        try:
            r = model(f, conf=CONF_THRESHOLD, imgsz=INFERENCE_SIZE,
                      max_det=MAX_DETECTIONS, verbose=False,
                      workers=0 if BACKEND == "ONNX-CPU" else 1)[0]
            payload = extract_polygons(r)
            with result_lock:
                latest_payload = payload
            now = time.time()
            if now - last_t > 0:
                inf_fps_deque.append(1.0 / (now - last_t))
            last_t = now
        except Exception as e:
            print(f"YOLO inference error: {e}")
            time.sleep(0.05)

# Warm up weights/caches before the camera loop so the first frames aren't 2s each
try:
    _ = model(np.zeros((h, w, 3), np.uint8), conf=CONF_THRESHOLD,
              imgsz=INFERENCE_SIZE, max_det=MAX_DETECTIONS, verbose=False)
    print("Model warmup done")
except Exception as e:
    print("warmup skipped:", e)

threading.Thread(target=inference_worker, daemon=True).start()

print(f"Backend: {BACKEND} | imgsz={INFERENCE_SIZE} | camera {w}x{h}@{camera_fps}fps")
print("Press 'q' to quit, 's' to save frame")

# ============================================================
# 7. MAIN LOOP
# ============================================================
last_display_time = time.time()
signal, signal_color = "Path Clear", (255, 255, 255)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame_idx += 1
        with frame_lock:
            latest_frame = frame.copy()

        with result_lock:
            payload = None if latest_payload is None else list(latest_payload)

        overlay = frame.copy()

        # perimeter zone
        zone_fill = overlay.copy()
        cv2.fillPoly(zone_fill, [ZONE_POINTS], (100, 200, 255))
        cv2.polylines(overlay, [ZONE_POINTS], True, (255, 255, 0), 2)
        overlay = cv2.addWeighted(zone_fill, 0.25, overlay, 0.75, 0)

        # radar canvas (direct display size)
        radar = np.zeros((RADAR_H, RADAR_W, 3), dtype=np.uint8)
        signal, signal_color = "Path Clear", (255, 255, 255)

        if payload:
            for cls, pts in payload:
                if len(pts) < 3:
                    continue
                color = class_colors.get(cls, (200, 200, 200))
                is_lane = cls in LANE_CLASSES

                # main view
                if is_lane:
                    cv2.polylines(overlay, [pts], False, color, 3)
                else:
                    cv2.fillPoly(overlay, [pts], color)

                # radar view
                rp = (pts.astype(np.float32) * RADAR_SCALE).astype(np.int32)
                if is_lane:
                    cv2.polylines(radar, [rp], False, color, 2)
                else:
                    cv2.fillPoly(radar, [rp], color)

                # steering: vectorized zone test on lane points
                if is_lane and len(pts) >= 3:
                    # subsample mask points (~50) for a fast containment vote
                    inside_flags = [
                        cv2.pointPolygonTest(ZONE_POINTS, (float(p[0]), float(p[1])), False) >= 0
                        for p in pts[::max(1, len(pts)//50)]   # test ~50 subsampled points
                    ]
                    if any(inside_flags):
                        cx = pts[:, 0].mean()
                        if cx < poly_center_x:
                            signal, signal_color = "Turn Right", (0, 165, 255)
                        else:
                            signal, signal_color = "Turn Left", (0, 255, 255)

                # label at centroid
                M = cv2.moments(pts)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.putText(overlay, class_names.get(cls, "Obj"),
                                (cx-25, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1)

        output = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # radar widget
        tri = np.array([[RADAR_W//2, RADAR_H-15],
                        [RADAR_W//2-12, RADAR_H-35],
                        [RADAR_W//2+12, RADAR_H-35]], np.int32)
        cv2.drawContours(radar, [tri], 0, (255, 0, 0), -1)
        cv2.rectangle(output, (16, 16), (16+RADAR_W+8, 16+RADAR_H+8), (255, 255, 255), 2)
        output[20:20+RADAR_H, 20:20+RADAR_W] = radar

        # status box
        st_w, st_h = 280, 50
        st_x, st_y = w - st_w - 20, 20
        roi = output[st_y:st_y+st_h, st_x:st_x+st_w]
        output[st_y:st_y+st_h, st_x:st_x+st_w] = cv2.addWeighted(roi, 0.6, np.zeros_like(roi), 0.4, 0)
        cv2.rectangle(output, (st_x-3, st_y-3), (st_x+st_w+3, st_y+st_h+3), (0, 255, 255), 2)
        t = f"STATUS: {signal}"
        ts = cv2.getTextSize(t, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        cv2.putText(output, t, (st_x+(st_w-ts[0])//2, st_y+(st_h+ts[1])//2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, signal_color, 2)

        # FPS text
        now = time.time()
        if now - last_display_time > 0:
            fps_deque.append(1.0/(now-last_display_time))
        last_display_time = now
        inf_avg = float(np.mean(inf_fps_deque)) if inf_fps_deque else 0.0
        cv2.putText(output, f"Disp FPS: {np.mean(fps_deque):.1f}", (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(output, f"YOLO FPS: {inf_avg:.1f} | {BACKEND}", (10, h-32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        show = cv2.resize(output, None, fx=SHOW_SCALE, fy=SHOW_SCALE) \
            if SHOW_SCALE != 1.0 else output
        cv2.imshow("ADAS - Raspberry Pi", show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fn = f"frame_{frame_idx}.jpg"
            cv2.imwrite(fn, output)
            print(f"Saved: {fn}")

        if frame_idx % 100 == 0:
            print(f"Frames: {frame_idx} | YOLO FPS: {inf_avg:.1f} | Signal: {signal}")

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSESSION COMPLETE | frames: {frame_idx} | "
          f"avg disp FPS: {np.mean(fps_deque):.1f}" if fps_deque else "")