"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           UNIFIED ADAS PIPELINE — run_pipeline.py                           ║
║  Models run in PARALLEL threads:                                             ║
║   1. Collision Prediction  (MobileNetV3 + LSTM)                              ║
║   2. Object Detection      (YOLOv8 — object_detection.pt)                   ║
║   3. Traffic Sign Detection(YOLOv8 — TrafficSign.pt)                         ║
║   4. Traffic Signal Det.   (YOLOv8 — traffic_signal.pt)                      ║
║   5. Lane Detection        (Classical Hough pipeline)                        ║
║   6. DMS / Driver Monitor  (YOLOv8 — dms2.pt)                               ║
║                                                                              ║
║  Outputs:                                                                    ║
║   • output_full.mp4    — full annotated video                                ║
║   • output_collision_clip.mp4 — 40 s clip (−20 s…+20 s around collision)    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python run_pipeline.py --video path/to/video.mp4

Optional:
    --output    output_full.mp4          (annotated full video)
    --clip      output_collision_clip.mp4 (40-sec collision clip)
    --high      0.33    collision high threshold
    --low       0.25    collision low threshold
"""

import argparse
import sys
import os
import cv2
import math
import time
import threading
import queue
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from collections import deque
from ultralytics import YOLO
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# EARLY SHIMS — inject before any model file is touched by pickle
# ─────────────────────────────────────────────────────────────────────────────
import sys as _sys, types as _types, importlib as _importlib
# numpy._core shim (numpy 1.x env loading model saved with numpy 2.x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "numpy._core" not in _sys.modules:
    try:
        import numpy.core as _nc, numpy.core.multiarray as _nma
        _cm = _types.ModuleType("numpy._core"); _cm.__package__ = "numpy._core"
        _mm = _types.ModuleType("numpy._core.multiarray"); _mm.__package__ = "numpy._core"
        [setattr(_cm, a, getattr(_nc,  a)) for a in dir(_nc)  if not a.startswith("__")]
        [setattr(_mm, a, getattr(_nma, a)) for a in dir(_nma) if not a.startswith("__")]
        _sys.modules["numpy._core"] = _cm
        _sys.modules["numpy._core.multiarray"] = _mm
    except Exception:
        pass
# DFLoss shim (ultralytics < 8.1 env loading model saved with >= 8.1)
for _lmod_name in ("ultralytics.utils.loss", "ultralytics.yolo.utils.loss"):
    try:
        _lmod = _importlib.import_module(_lmod_name)
        if not hasattr(_lmod, "DFLoss"):
            import torch.nn as _nn
            class _DFLoss(_nn.Module):
                def __init__(self, reg_max=16): super().__init__(); self.reg_max = reg_max
                def forward(self, p, t):
                    import torch.nn.functional as F; tl=t.long(); tr=tl+1; wl=tr.float()-t; wr=1-wl
                    return (F.cross_entropy(p,tl.clamp(0,self.reg_max-1),reduction="none")*wl
                           +F.cross_entropy(p,tr.clamp(0,self.reg_max-1),reduction="none")*wr).mean(-1,keepdim=True)
            _lmod.DFLoss = _DFLoss
        break
    except ModuleNotFoundError:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

MODEL_COLLISION   = os.path.join(MODELS_DIR, "MobileNetv3_collisionmodel.pth")
MODEL_OBJ_DET     = os.path.join(MODELS_DIR, "object_detection.pt")
MODEL_TRAFFIC_SIGN= os.path.join(MODELS_DIR, "TrafficSign.pt")
MODEL_TRAFFIC_SIG = os.path.join(MODELS_DIR, "traffic_signal.pt")
MODEL_DMS         = os.path.join(MODELS_DIR, "dms2.pt")

# ─────────────────────────────────────────────────────────────────────────────
# COLLISION PREDICTOR ARCHITECTURE  (must match training)
# ─────────────────────────────────────────────────────────────────────────────
class CollisionPredictor(nn.Module):
    def __init__(self, num_frames=16, lstm_hidden=256):
        super().__init__()
        self.backbone = mobilenet_v3_large(weights=None)
        backbone_out = self.backbone.classifier[0].in_features  # 960
        self.backbone.classifier = nn.Identity()
        self.lstm = nn.LSTM(backbone_out, lstm_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.backbone(x).view(b, t, -1)
        lstm_out, _ = self.lstm(features)
        return self.classifier(lstm_out[:, -1, :]).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# LANE DETECTION  (classical Hough)
# ─────────────────────────────────────────────────────────────────────────────
def _roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def _draw_lanes(img, left_line, right_line):
    overlay = np.zeros_like(img)
    pts = np.array([[
        (left_line[0],  left_line[1]),
        (left_line[2],  left_line[3]),
        (right_line[2], right_line[3]),
        (right_line[0], right_line[1]),
    ]], dtype=np.int32)
    cv2.fillPoly(overlay, pts, (0, 200, 0))
    return cv2.addWeighted(img, 1.0, overlay, 0.35, 0.0)


def detect_lanes(frame_bgr):
    """Returns (annotated_frame, departure_flag)"""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    vertices = np.array([[(0, h), (w // 2, h // 2), (w, h)]], np.int32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    masked = _roi(edges, vertices)

    lines = cv2.HoughLinesP(masked, 6, np.pi / 180, 120,
                             minLineLength=30, maxLineGap=100)
    if lines is None:
        return frame_bgr, False

    lx, ly, rx, ry = [], [], [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            if abs(slope) < 0.5:
                continue
            if slope <= 0:
                lx += [x1, x2]; ly += [y1, y2]
            else:
                rx += [x1, x2]; ry += [y1, y2]

    min_y = int(h * 0.6)
    max_y = h

    def fit(xs, ys):
        if xs and ys:
            p = np.poly1d(np.polyfit(ys, xs, 1))
            return (int(p(max_y)), max_y, int(p(min_y)), min_y)
        return None

    left  = fit(lx, ly)
    right = fit(rx, ry)

    out = frame_bgr.copy()
    departure = False

    if left and right:
        out = _draw_lanes(out, left, right)
        # departure: lane centre vs frame centre
        lane_cx = (left[0] + right[0]) // 2
        frame_cx = w // 2
        if abs(lane_cx - frame_cx) > w * 0.12:
            departure = True
            cv2.putText(out, "⚠ LANE DEPARTURE", (w // 2 - 140, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    elif left:
        cv2.line(out, (left[0], left[1]), (left[2], left[3]), (0, 255, 0), 4)
    elif right:
        cv2.line(out, (right[0], right[1]), (right[2], right[3]), (0, 255, 0), 4)

    return out, departure


# ─────────────────────────────────────────────────────────────────────────────
# PROXIMITY BAR  (from eye_sleep logic — no winsound)
# ─────────────────────────────────────────────────────────────────────────────
_prev_proximity = 0

def draw_proximity_hud(frame, yolo_results):
    """Draw trapezoid proximity HUD (from eye_sleep.py logic)."""
    global _prev_proximity
    h, w = frame.shape[:2]
    highest = 0

    for box in yolo_results[0].boxes:
        cls = int(box.cls[0])
        if cls in [0, 2, 3, 5, 7]:
            _, _, _, y2 = map(int, box.xyxy[0])
            if   y2 > h * 0.88: prox = 3
            elif y2 > h * 0.72: prox = 2
            elif y2 > h * 0.58: prox = 1
            else:                prox = 0
            highest = max(highest, prox)

    # smooth
    if highest > _prev_proximity:
        cur = highest
    else:
        cur = int(_prev_proximity * 0.7 + highest * 0.3)
    _prev_proximity = cur

    color_map = {3: (0,0,255), 2: (0,255,255), 1: (222,222,0), 0: (0,200,0)}
    label_map = {3: "STOP!", 2: "CAUTION: CLOSE", 1: "APPROACHING", 0: "CLEAR"}
    color = color_map[min(cur, 3)]
    label = label_map[min(cur, 3)]

    pts = np.array([
        [int(w*0.38), int(h*0.55)],
        [int(w*0.62), int(h*0.55)],
        [int(w*0.95), int(h*0.85)],
        [int(w*0.05), int(h*0.85)],
    ], np.int32)
    cv2.polylines(frame, [pts], True, color, 2)
    cv2.line(frame, tuple(pts[3]), tuple(pts[2]), color, 6)
    lm = ((pts[0][0]+pts[3][0])//2, (pts[0][1]+pts[3][1])//2)
    rm = ((pts[1][0]+pts[2][0])//2, (pts[1][1]+pts[2][1])//2)
    cv2.line(frame, lm, rm, color, 2)
    cv2.putText(frame, label, (int(w*0.38), int(h*0.53)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# THREAD WORKER: run YOLO model on a frame, put result in result_dict
# ─────────────────────────────────────────────────────────────────────────────
def _yolo_worker(model, frame, key, result_dict):
    result_dict[key] = model(frame, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "obj":   (0, 255, 0),
    "sign":  (255, 128, 0),
    "sig":   (0, 200, 255),
    "dms":   (200, 0, 255),
}

def _draw_yolo(frame, results, color, tag, conf_thresh=0.35):
    if results is None:
        return
    for box in results[0].boxes:
        if float(box.conf[0]) < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        names  = results[0].names
        label  = f"{tag}:{names[cls_id]} {box.conf[0]:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def _draw_collision_overlay(frame, ema_score, alert_active):
    h, w = frame.shape[:2]
    bar_w = int(w * 0.25)
    bar_h = 16
    bx, by = 10, h - 30
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (60, 60, 60), -1)
    fill = int(bar_w * min(ema_score, 1.0))
    bar_color = (0, 0, 255) if alert_active else (0, 200, 255)
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (200, 200, 200), 1)
    cv2.putText(frame, f"Collision:{ema_score:.2f}", (bx, by - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if alert_active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
        frame[:] = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
        cv2.putText(frame, "⚠ COLLISION DETECTED", (w // 2 - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


def _draw_status_bar(frame, fps_disp, frame_idx, total, collision_time):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
    ts = frame_idx / max(fps_disp, 1)
    col_str = f"  |  Collision@{collision_time:.1f}s" if collision_time >= 0 else ""
    text = (f"Frame {frame_idx}/{total}  t={ts:.1f}s  FPS≈{fps_disp:.0f}{col_str}")
    cv2.putText(frame, text, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1)


# ─────────────────────────────────────────────────────────────────────────────
# COMPATIBILITY SHIMS — must run BEFORE torch.load / pickle deserializes dms2.pt
# Fixes two issues with dms2.pt saved on a newer environment:
#   1. numpy._core missing  (model saved with numpy>=2.0, env has numpy<2.0)
#   2. DFLoss missing       (model saved with ultralytics>=8.1, env older)
# Both shims inject fake modules/attributes into sys.modules so pickle finds them.
# ─────────────────────────────────────────────────────────────────────────────
def _install_compat_shims():
    import sys, types, importlib
    import torch.nn as nn

    # ── 1. numpy._core  (numpy 1.x → 2.x bridge) ────────────────────────────
    import numpy as np
    if "numpy._core" not in sys.modules:
        import numpy.core as _nc
        _core_mod = types.ModuleType("numpy._core")
        _core_mod.__package__ = "numpy._core"
        for attr in dir(_nc):
            try:
                setattr(_core_mod, attr, getattr(_nc, attr))
            except Exception:
                pass
        sys.modules["numpy._core"] = _core_mod

        _ma_mod = types.ModuleType("numpy._core.multiarray")
        _ma_mod.__package__ = "numpy._core"
        import numpy.core.multiarray as _nma
        for attr in dir(_nma):
            try:
                setattr(_ma_mod, attr, getattr(_nma, attr))
            except Exception:
                pass
        sys.modules["numpy._core.multiarray"] = _ma_mod
        print("  ⚠  numpy._core shim injected (numpy 1.x / 2.x mismatch)")

    # ── 2. DFLoss  (ultralytics < 8.1 shim) ──────────────────────────────────
    for mod_name in ("ultralytics.utils.loss", "ultralytics.yolo.utils.loss"):
        try:
            loss_mod = importlib.import_module(mod_name)
            break
        except ModuleNotFoundError:
            loss_mod = None

    if loss_mod is not None and not hasattr(loss_mod, "DFLoss"):
        class DFLoss(nn.Module):
            """Compatibility shim — DFLoss introduced in ultralytics >= 8.1."""
            def __init__(self, reg_max=16):
                super().__init__()
                self.reg_max = reg_max
            def forward(self, pred_dist, target):
                import torch.nn.functional as F
                tl = target.long()
                tr = tl + 1
                wl = tr.float() - target
                wr = 1.0 - wl
                return (
                    F.cross_entropy(pred_dist, tl.clamp(0, self.reg_max - 1),
                                    reduction="none") * wl
                    + F.cross_entropy(pred_dist, tr.clamp(0, self.reg_max - 1),
                                      reduction="none") * wr
                ).mean(-1, keepdim=True)
        loss_mod.DFLoss = DFLoss
        print("  ⚠  DFLoss shim injected (ultralytics version mismatch)")


def _load_dms_model(model_path, device):
    """Install all compatibility shims, then load dms2.pt."""
    _install_compat_shims()
    return YOLO(model_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run(video_path, output_path, clip_path,
        high_thresh=0.33, low_thresh=0.25,
        stride=4, window_frames=16,
        min_confirm=8, hold_calls=5,
        clip_before=20.0, clip_after=20.0):

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  UNIFIED ADAS PIPELINE")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    # ── Load models ──────────────────────────────────────────────────────────
    print("  Loading models …")

    # Collision
    col_model = CollisionPredictor(lstm_hidden=256).to(device)
    col_model.load_state_dict(torch.load(MODEL_COLLISION, map_location=device,
                                          weights_only=False))
    col_model.eval()
    col_transform = Compose([
        Resize(224, 224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    print("  ✓ Collision model")

    # YOLO models  (thread-safe: each call returns new Results, models are stateless at inference)
    yolo_obj   = YOLO(MODEL_OBJ_DET)
    yolo_sign  = YOLO(MODEL_TRAFFIC_SIGN)
    yolo_sig   = YOLO(MODEL_TRAFFIC_SIG)
    yolo_dms   = _load_dms_model(MODEL_DMS, device)
    print("  ✓ YOLO models (obj / sign / signal / dms)")

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video  : {W}×{H} @ {fps:.1f} fps | {total} frames "
          f"({total/fps:.1f} s)")

    # ── Output writer ────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w  = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # ── Collision state ──────────────────────────────────────────────────────
    frame_ring      = deque(maxlen=window_frames)
    ema_score       = 0.0
    ema_alpha       = 0.4
    consecutive_hot = 0
    alert_active    = False
    hold_countdown  = 0
    prev_gray       = None
    collision_frame = -1   # frame index when first confirmed
    collision_time  = -1.0

    # ── Per-frame score log (for clip extraction) ────────────────────────────
    all_scores      = []   # list of (frame_idx, ema_score)

    frame_idx = 0
    t0        = time.time()

    pbar = tqdm(total=total, desc="Processing", unit="fr",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_ring.append(frame_bgr.copy())

        # ── 1. Lane detection (fast, no thread needed) ────────────────────
        frame_bgr, _lane_dep = detect_lanes(frame_bgr)

        # ── 2. Parallel YOLO inference ────────────────────────────────────
        yolo_results = {}
        threads = []
        for model, key in [(yolo_obj, "obj"), (yolo_sign, "sign"),
                           (yolo_sig, "sig"),  (yolo_dms,  "dms")]:
            t = threading.Thread(target=_yolo_worker,
                                 args=(model, frame_bgr.copy(), key, yolo_results))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # Draw YOLO annotations
        _draw_yolo(frame_bgr, yolo_results.get("obj"),  COLORS["obj"],  "OBJ")
        _draw_yolo(frame_bgr, yolo_results.get("sign"), COLORS["sign"], "SIGN")
        _draw_yolo(frame_bgr, yolo_results.get("sig"),  COLORS["sig"],  "SIG")
        _draw_yolo(frame_bgr, yolo_results.get("dms"),  COLORS["dms"],  "DMS")

        # Proximity HUD
        if yolo_results.get("obj"):
            draw_proximity_hud(frame_bgr, yolo_results["obj"])

        # ── 3. Collision inference (every `stride` frames) ────────────────
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # scene filter
        flow_mag = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_mag = float(np.mean(np.linalg.norm(flow, axis=2)))
        prev_gray = gray

        edges        = cv2.Canny(gray, 60, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        scene_blocked = (ema_score < 0.38
                         and flow_mag < 0.8
                         and edge_density > 0.25)

        if frame_idx % stride == 0 and len(frame_ring) == window_frames:
            if not scene_blocked:
                clip_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                            for f in frame_ring]
                tensor = torch.stack(
                    [col_transform(image=img)["image"] for img in clip_rgb]
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    raw = torch.sigmoid(col_model(tensor)).item()
                ema_score = ema_alpha * raw + (1 - ema_alpha) * ema_score
            else:
                ema_score *= 0.92

            if ema_score >= high_thresh:
                consecutive_hot += 1
            else:
                consecutive_hot = max(0, consecutive_hot - 1)

            if not alert_active:
                if consecutive_hot >= min_confirm:
                    alert_active    = True
                    hold_countdown  = hold_calls
                    if collision_frame < 0:
                        collision_frame = frame_idx
                        collision_time  = frame_idx / fps
                        print(f"\n  🚨 Collision confirmed @ frame {frame_idx} "
                              f"(t={collision_time:.1f}s, score={ema_score:.3f})")
            else:
                if ema_score < low_thresh:
                    hold_countdown -= 1
                    if hold_countdown <= 0:
                        alert_active    = False
                        consecutive_hot = 0
                else:
                    hold_countdown = hold_calls

        all_scores.append((frame_idx, ema_score))

        # ── 4. Collision HUD & status bar ─────────────────────────────────
        _draw_collision_overlay(frame_bgr, ema_score, alert_active)
        _draw_status_bar(frame_bgr, fps, frame_idx, total, collision_time)

        out_w.write(frame_bgr)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out_w.release()
    elapsed = time.time() - t0
    print(f"\n  ✓ Full annotated video saved → {output_path}")
    print(f"    Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx/elapsed:.1f} fr/s)")

    # ── 5. Extract 40-second collision clip ──────────────────────────────────
    if collision_frame >= 0:
        _extract_clip(video_path, clip_path, collision_time, fps, total,
                      clip_before, clip_after)
    else:
        print("\n  ℹ No collision detected — skipping clip extraction.")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  DONE")
    if collision_frame >= 0:
        print(f"  Collision time : {collision_time:.1f}s")
        print(f"  Collision clip : {clip_path}")
    print(f"  Full video     : {output_path}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLIP EXTRACTION  (−20 s … +20 s around collision)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_clip(video_path, clip_path, collision_time_s, fps, total_frames,
                  before=20.0, after=20.0):
    duration = total_frames / fps
    t_start  = max(0.0,        collision_time_s - before)
    t_end    = min(duration,   collision_time_s + after)

    f_start  = int(t_start * fps)
    f_end    = int(t_end   * fps)

    print(f"\n  Extracting collision clip: {t_start:.1f}s → {t_end:.1f}s "
          f"(frames {f_start}–{f_end})")

    cap = cv2.VideoCapture(video_path)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    clip_w = cv2.VideoWriter(clip_path, fourcc, fps, (W, H))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    collision_rel = collision_time_s - t_start   # seconds into the clip

    for fi in range(f_end - f_start):
        ret, frame = cap.read()
        if not ret:
            break

        t_in_clip = t_start + fi / fps
        t_from_col = t_in_clip - collision_time_s

        # Phase label
        if   t_from_col < -2:  phase, pcol = "PRE-COLLISION",  (0, 200, 0)
        elif t_from_col < 2:   phase, pcol = "AT COLLISION",   (0, 0, 255)
        else:                  phase, pcol = "POST-COLLISION",  (0, 140, 255)

        # Red flash at collision moment
        if abs(t_from_col) < 2:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 200), -1)
            frame = cv2.addWeighted(ov, 0.20, frame, 0.80, 0)

        # Overlay
        cv2.rectangle(frame, (0, 0), (W, 38), (10, 10, 10), -1)
        cv2.putText(frame, phase, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, pcol, 2)
        cv2.putText(frame,
                    f"t={t_in_clip:.1f}s  ({t_from_col:+.1f}s from collision)",
                    (240, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1)

        # Progress bar
        prog = fi / max(f_end - f_start - 1, 1)
        bar_w = W - 20
        cv2.rectangle(frame, (10, H-14), (10 + bar_w, H-6), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, H-14), (10 + int(bar_w * prog), H-6), pcol, -1)
        # Collision marker
        col_x = 10 + int(bar_w * (collision_rel / (t_end - t_start)))
        cv2.line(frame, (col_x, H-18), (col_x, H-2), (0, 0, 255), 2)

        clip_w.write(frame)

    cap.release()
    clip_w.release()
    print(f"  ✓ Collision clip saved → {clip_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Unified ADAS Pipeline — all models in parallel")
    ap.add_argument("--video",  required=True,
                    help="Path to input video file")
    ap.add_argument("--output", default="output_full.mp4",
                    help="Output annotated video (default: output_full.mp4)")
    ap.add_argument("--clip",   default="output_collision_clip.mp4",
                    help="40-sec collision clip (default: output_collision_clip.mp4)")
    ap.add_argument("--high",   type=float, default=0.33,
                    help="Collision high threshold (default: 0.33)")
    ap.add_argument("--low",    type=float, default=0.25,
                    help="Collision low threshold (default: 0.25)")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"Video not found: {args.video}")

    run(
        video_path  = args.video,
        output_path = args.output,
        clip_path   = args.clip,
        high_thresh = args.high,
        low_thresh  = args.low,
    )


if __name__ == "__main__":
    main()