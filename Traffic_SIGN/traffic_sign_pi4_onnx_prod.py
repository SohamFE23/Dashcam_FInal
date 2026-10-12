#!/usr/bin/env python3
"""
Raspberry Pi 4 production traffic-light detector (YOLO11 ONNX)

- Picamera2 640x480 @ 25 FPS
- ONNX Runtime CPU inference @ 320x320
- No torch, no ultralytics, no NCNN
- Latest-frame worker thread (no backlog)
- Headless by default (safe over SSH / Wayland)
- Optional GUI:    --gui
- Optional record: --record output.mp4

Classes:
0 = green
1 = red
2 = yellow
"""

import argparse
import os
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
from libcamera import controls

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "best.onnx"

CAM_W, CAM_H = 640, 480
CAM_FPS = 25
IMG_SIZE = 320
CONF_THRES = 0.35
IOU_THRES = 0.45

CLASS_NAMES = {0: "green", 1: "red", 2: "yellow"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 255, 255)}


def set_affinity(cpus):
    try:
        os.sched_setaffinity(0, set(cpus))
    except Exception:
        pass


def letterbox_rgb(img, size=320):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def preprocess(img_rgb):
    img, scale, px, py = letterbox_rgb(img_rgb, IMG_SIZE)
    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(x), scale, px, py


def iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih

    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-7)


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[iou_one_to_many(boxes[i], boxes[rest]) <= iou_thres]
    return keep


def normalize_output(raw):
    a = np.asarray(raw, dtype=np.float32)
    a = np.squeeze(a)

    if a.ndim != 2:
        raise RuntimeError(f"Unexpected output shape: {a.shape}")

    # YOLO11 detect with 3 classes => 4 box values + 3 class scores = 7
    if a.shape[0] == 7:
        return a.T
    if a.shape[1] == 7:
        return a

    raise RuntimeError(f"Expected (7,N) or (N,7), got {a.shape}")


def decode(raw, scale, px, py, fw, fh):
    p = normalize_output(raw)

    cls_scores = p[:, 4:7]
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(len(p)), cls_ids]

    good = scores >= CONF_THRES
    if not np.any(good):
        return []

    p = p[good]
    scores = scores[good]
    cls_ids = cls_ids[good]

    cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]

    boxes = np.empty((len(p), 4), dtype=np.float32)
    boxes[:, 0] = (cx - bw / 2 - px) / scale
    boxes[:, 1] = (cy - bh / 2 - py) / scale
    boxes[:, 2] = (cx + bw / 2 - px) / scale
    boxes[:, 3] = (cy + bh / 2 - py) / scale

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, fw - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, fh - 1)

    out = []
    for cid in np.unique(cls_ids):
        idx = np.where(cls_ids == cid)[0]
        for k in nms(boxes[idx], scores[idx], IOU_THRES):
            j = idx[k]
            out.append((int(cls_ids[j]), float(scores[j]), boxes[j].copy()))

    out.sort(key=lambda x: x[1], reverse=True)
    return out


class Detector:
    def __init__(self, model_path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 3
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"ONNX Runtime: {ort.__version__}", flush=True)
        print(f"Input : {self.input_name} {self.session.get_inputs()[0].shape}", flush=True)
        print(f"Output: {self.output_name} {self.session.get_outputs()[0].shape}", flush=True)

        dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for _ in range(2):
            self.session.run([self.output_name], {self.input_name: dummy})

    def infer(self, rgb):
        x, scale, px, py = preprocess(rgb)
        t0 = time.perf_counter()
        raw = self.session.run([self.output_name], {self.input_name: x})[0]
        ms = (time.perf_counter() - t0) * 1000.0
        dets = decode(raw, scale, px, py, rgb.shape[1], rgb.shape[0])
        return dets, ms


def draw(frame, dets):
    for cid, score, box in dets:
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS.get(cid, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cid, cid)} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--gui", action="store_true",
                    help="Show OpenCV window. Use from the Pi desktop, not normal SSH.")
    ap.add_argument("--record", default="",
                    help="Optional annotated MP4 output path")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(2)

    # Keep camera/main loop on core 0.
    set_affinity([0])

    stop = threading.Event()
    ready = threading.Event()
    frame_lock = threading.Lock()
    result_lock = threading.Lock()

    latest = {"seq": -1, "rgb": None}
    result = {"dets": [], "infer_ms": 0.0, "error": None, "infer_count": 0}

    def worker():
        # Create ORT after setting worker affinity.
        set_affinity([1, 2, 3])
        try:
            detector = Detector(model_path)
        except Exception as e:
            with result_lock:
                result["error"] = f"Detector init: {e}"
            ready.set()
            stop.set()
            return

        ready.set()
        seen = -1

        while not stop.is_set():
            with frame_lock:
                seq = latest["seq"]
                rgb = latest["rgb"]

            if rgb is None or seq == seen:
                time.sleep(0.001)
                continue

            seen = seq
            try:
                dets, ms = detector.infer(rgb)
                with result_lock:
                    result["dets"] = dets
                    result["infer_ms"] = ms
                    result["infer_count"] += 1
            except Exception as e:
                with result_lock:
                    result["error"] = f"Inference: {e}"
                stop.set()

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    ready.wait()

    with result_lock:
        if result["error"]:
            raise RuntimeError(result["error"])

    # Start camera only after model initialization succeeds.
    picam2 = Picamera2()
    from libcamera import controls

    config = picam2.create_video_configuration(
        main={
            "size": (CAM_W, CAM_H),
            "format": "RGB888"
        },
        controls={
            "FrameRate": CAM_FPS,

            # Better indoor colour balance
            "AwbMode": controls.AwbModeEnum.Fluorescent,

            # Image tuning
            "Brightness": 0.05,
            "Contrast": 1.05,
            "Saturation": 0.95,
            "Sharpness": 1.2,

            # Slight exposure lift
            "ExposureValue": 0.3,
        },
        buffer_count=4,
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, CAM_FPS, (CAM_W, CAM_H))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open recorder: {args.record}")

    seq = 0
    capture_count = 0
    last_report = time.perf_counter()
    last_infer_count = 0

    print("RUNNING. Ctrl+C to stop.", flush=True)
    if args.gui:
        print("GUI enabled. Press q or ESC to quit.", flush=True)
    else:
        print("Headless mode enabled (safe for SSH/Wayland).", flush=True)

    try:
        while not stop.is_set():
            frame = picam2.capture_array()
            rgb = frame[:, :, ::-1].copy()

            with frame_lock:
                latest["seq"] = seq
                latest["rgb"] = rgb
            seq += 1
            capture_count += 1

            need_frame = args.gui or writer is not None
            if need_frame:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                with result_lock:
                    dets = list(result["dets"])
                    infer_ms = float(result["infer_ms"])
                draw(bgr, dets)

                infer_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
                cv2.putText(
                    bgr,
                    f"ORT {infer_fps:.1f} FPS | {infer_ms:.0f} ms",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if writer is not None:
                    writer.write(bgr)

                if args.gui:
                    cv2.imshow("Traffic Sign - Pi4 ONNX", bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

            now = time.perf_counter()
            if now - last_report >= 2.0:
                dt = now - last_report

                with result_lock:
                    infer_count = result["infer_count"]
                    infer_ms = float(result["infer_ms"])
                    dets = list(result["dets"])
                    err = result["error"]

                cap_fps = capture_count / dt
                inf_fps = (infer_count - last_infer_count) / dt
                names = ", ".join(
                    f"{CLASS_NAMES[c]}:{s:.2f}" for c, s, _ in dets[:5]
                ) or "none"

                print(
                    f"camera={cap_fps:.1f} FPS | "
                    f"inference={inf_fps:.1f} FPS | "
                    f"last={infer_ms:.0f} ms | "
                    f"detections={names}",
                    flush=True,
                )

                if err:
                    print(f"ERROR: {err}", file=sys.stderr, flush=True)

                capture_count = 0
                last_infer_count = infer_count
                last_report = now

    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        stop.set()
        worker_thread.join(timeout=2.0)
        try:
            picam2.stop()
        except Exception:
            pass
        if writer is not None:
            writer.release()
        if args.gui:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()
