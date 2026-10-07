#!/usr/bin/env python3
"""
Stable Raspberry Pi 4 traffic-light detector using ONNX Runtime.
Runtime dependencies: picamera2, opencv-python, numpy, onnxruntime
No torch. No ultralytics. No ncnn.

Expected model:
  best.onnx
  input:  1x3x320x320
  output: YOLO11 detect tensor, typically (1, 7, N) for 3 classes
classes:
  0 green
  1 red
  2 yellow
"""

import os
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "best.onnx"

CAM_W, CAM_H = 640, 480
CAM_FPS = 25
IMG_SIZE = 320

CONF_THRES = 0.35
IOU_THRES = 0.45

CLASS_NAMES = {0: "green", 1: "red", 2: "yellow"}
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (0, 255, 255),
}


def set_affinity(cpus):
    try:
        os.sched_setaffinity(0, set(cpus))
    except Exception:
        pass


def letterbox_rgb(img_rgb, size=320):
    h, w = img_rgb.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized

    return canvas, scale, left, top


def preprocess(img_rgb):
    img, scale, pad_x, pad_y = letterbox_rgb(img_rgb, IMG_SIZE)
    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0)
    return np.ascontiguousarray(x), scale, pad_x, pad_y


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
        ious = iou_one_to_many(boxes[i], boxes[rest])
        order = rest[ious <= iou_thres]

    return keep


def normalize_output(raw):
    arr = np.asarray(raw, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected YOLO output shape: {arr.shape}")

    # 3 classes => 4 box values + 3 class scores = 7
    if arr.shape[0] == 7:
        arr = arr.T
    elif arr.shape[1] == 7:
        pass
    else:
        raise RuntimeError(
            f"Unexpected output shape {arr.shape}; expected (7,N) or (N,7)"
        )

    return arr


def decode(raw, scale, pad_x, pad_y, frame_w, frame_h):
    pred = normalize_output(raw)

    class_scores = pred[:, 4:7]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(pred)), class_ids]

    valid = scores >= CONF_THRES
    if not np.any(valid):
        return []

    p = pred[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]

    boxes = np.empty((len(p), 4), dtype=np.float32)
    boxes[:, 0] = (cx - bw / 2 - pad_x) / scale
    boxes[:, 1] = (cy - bh / 2 - pad_y) / scale
    boxes[:, 2] = (cx + bw / 2 - pad_x) / scale
    boxes[:, 3] = (cy + bh / 2 - pad_y) / scale

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_h - 1)

    detections = []

    for cls_id in np.unique(class_ids):
        idx = np.where(class_ids == cls_id)[0]
        for k in nms(boxes[idx], scores[idx], IOU_THRES):
            j = idx[k]
            detections.append(
                (int(class_ids[j]), float(scores[j]), boxes[j].copy())
            )

    detections.sort(key=lambda x: x[1], reverse=True)
    return detections


class ORTDetector:
    def __init__(self):
        if not MODEL.is_file():
            raise FileNotFoundError(f"Missing model: {MODEL}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 3
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(MODEL),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("ONNX Runtime:", ort.__version__)
        print("Input :", self.input_name, self.session.get_inputs()[0].shape)
        print("Output:", self.output_name, self.session.get_outputs()[0].shape)

        # Warm-up
        dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for _ in range(2):
            self.session.run([self.output_name], {self.input_name: dummy})

    def infer(self, img_rgb):
        x, scale, px, py = preprocess(img_rgb)

        t0 = time.perf_counter()
        out = self.session.run(
            [self.output_name],
            {self.input_name: x},
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        dets = decode(
            out, scale, px, py,
            img_rgb.shape[1], img_rgb.shape[0]
        )
        return dets, infer_ms


def draw(frame, detections):
    for cls_id, score, box in detections:
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
        )


def main():
    # main/camera/UI on core 0 when possible
    set_affinity([0])

    detector_ready = threading.Event()
    stop = threading.Event()
    frame_lock = threading.Lock()
    result_lock = threading.Lock()

    latest = {"seq": -1, "rgb": None}
    result = {
        "seq": -1,
        "dets": [],
        "infer_ms": 0.0,
        "error": None,
    }

    def worker():
        set_affinity([1, 2, 3])

        try:
            detector = ORTDetector()
            detector_ready.set()
        except Exception as e:
            with result_lock:
                result["error"] = f"Model init failed: {e}"
            detector_ready.set()
            stop.set()
            return

        seen = -1

        while not stop.is_set():
            with frame_lock:
                seq = latest["seq"]
                frame = latest["rgb"]

            if frame is None or seq == seen:
                time.sleep(0.001)
                continue

            seen = seq

            try:
                dets, infer_ms = detector.infer(frame)
                with result_lock:
                    result["seq"] = seq
                    result["dets"] = dets
                    result["infer_ms"] = infer_ms
                    result["error"] = None
            except Exception as e:
                with result_lock:
                    result["error"] = str(e)
                stop.set()
                return

    thread = threading.Thread(target=worker, name="ort-worker", daemon=True)
    thread.start()

    # Wait for model initialization before opening camera.
    detector_ready.wait()

    with result_lock:
        if result["error"]:
            raise RuntimeError(result["error"])

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (CAM_W, CAM_H), "format": "RGB888"},
        controls={"FrameRate": CAM_FPS},
        buffer_count=4,
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    seq = 0
    frames = 0
    display_fps = 0.0
    fps_t0 = time.perf_counter()

    try:
        while not stop.is_set():
            rgb = picam2.capture_array()

            with frame_lock:
                latest["seq"] = seq
                latest["rgb"] = rgb
            seq += 1

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            with result_lock:
                dets = list(result["dets"])
                infer_ms = float(result["infer_ms"])
                err = result["error"]

            draw(bgr, dets)

            frames += 1
            now = time.perf_counter()
            if now - fps_t0 >= 1.0:
                display_fps = frames / (now - fps_t0)
                frames = 0
                fps_t0 = now

            infer_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
            status = (
                f"Display {display_fps:.1f} FPS | "
                f"ORT {infer_fps:.1f} FPS | {infer_ms:.0f} ms"
            )
            cv2.putText(
                bgr, status, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            if err:
                cv2.putText(
                    bgr, err[:100], (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 2, cv2.LINE_AA
                )

            cv2.imshow("Traffic Sign - ONNX Runtime", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        stop.set()
        thread.join(timeout=2.0)
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
