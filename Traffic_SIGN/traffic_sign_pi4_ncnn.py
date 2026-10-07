#!/usr/bin/env python3
"""
Production Raspberry Pi 4 traffic-light detector
- Ultralytics YOLO11 detect model exported to NCNN at 320x320
- Native ncnn Python runtime (no torch, no ultralytics at runtime)
- Picamera2 640x480 @ 25 FPS
- Latest-frame worker thread, no inference backlog
- Classes from metadata: 0=green, 1=red, 2=yellow

IMPORTANT:
If NCNN segfaults inside ex.extract("out0"), the exported NCNN model is broken/incompatible.
Re-export with the known-good PNNX version before using this application.
"""

import os
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import ncnn
from picamera2 import Picamera2

MODEL_DIR = Path(__file__).resolve().parent / "best_ncnn_model"
PARAM = str(MODEL_DIR / "model.ncnn.param")
BIN = str(MODEL_DIR / "model.ncnn.bin")

CAM_W, CAM_H = 640, 480
CAM_FPS = 25
IMG_SIZE = 320

CONF_THRES = 0.35
IOU_THRES = 0.45
NCNN_THREADS = 3

CLASS_NAMES = {0: "green", 1: "red", 2: "yellow"}

# OpenCV BGR colors
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


def box_iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih

    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
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
        ious = box_iou_one_to_many(boxes[i], boxes[rest])
        order = rest[ious <= iou_thres]

    return keep


def normalize_output(out):
    arr = np.asarray(out.numpy(), dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected NCNN output shape: {arr.shape}")

    # YOLO11 detect, nc=3 => each prediction has 7 values: xywh + 3 classes.
    if arr.shape[0] == 7:
        arr = arr.T
    elif arr.shape[1] == 7:
        pass
    else:
        raise RuntimeError(
            f"Unexpected output shape {arr.shape}; expected (7,N) or (N,7) "
            "for a 3-class YOLO11 detect model."
        )

    return arr


def decode_predictions(out, scale, pad_x, pad_y, frame_w, frame_h):
    pred = normalize_output(out)

    cls_scores = pred[:, 4:7]
    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(len(pred)), class_ids]

    mask = scores >= CONF_THRES
    if not np.any(mask):
        return []

    p = pred[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    cx = p[:, 0]
    cy = p[:, 1]
    bw = p[:, 2]
    bh = p[:, 3]

    boxes = np.empty((len(p), 4), dtype=np.float32)
    boxes[:, 0] = (cx - bw / 2 - pad_x) / scale
    boxes[:, 1] = (cy - bh / 2 - pad_y) / scale
    boxes[:, 2] = (cx + bw / 2 - pad_x) / scale
    boxes[:, 3] = (cy + bh / 2 - pad_y) / scale

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_h - 1)

    detections = []

    # Class-aware NMS
    for cls_id in np.unique(class_ids):
        idx = np.where(class_ids == cls_id)[0]
        keep = nms(boxes[idx], scores[idx], IOU_THRES)

        for k in keep:
            j = idx[k]
            detections.append(
                (
                    int(class_ids[j]),
                    float(scores[j]),
                    boxes[j].copy(),
                )
            )

    detections.sort(key=lambda x: x[1], reverse=True)
    return detections


class NCNNDetector:
    def __init__(self):
        if not Path(PARAM).is_file() or not Path(BIN).is_file():
            raise FileNotFoundError(
                f"NCNN model files not found in {MODEL_DIR}"
            )

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = NCNN_THREADS

        r1 = self.net.load_param(PARAM)
        r2 = self.net.load_model(BIN)
        if r1 != 0 or r2 != 0:
            raise RuntimeError(f"NCNN model load failed: param={r1}, bin={r2}")

    def infer(self, img_rgb):
        inp, scale, px, py = letterbox_rgb(img_rgb, IMG_SIZE)

        mat = ncnn.Mat.from_pixels(
            inp,
            ncnn.Mat.PixelType.PIXEL_RGB,
            IMG_SIZE,
            IMG_SIZE,
        )
        mat.substract_mean_normalize(
            [],
            [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],
        )

        ex = self.net.create_extractor()
        ret = ex.input("in0", mat)
        if ret != 0:
            raise RuntimeError(f"NCNN input failed: {ret}")

        t0 = time.perf_counter()
        ret, out = ex.extract("out0")
        infer_ms = (time.perf_counter() - t0) * 1000.0

        if ret != 0:
            raise RuntimeError(f"NCNN extract failed: {ret}")

        dets = decode_predictions(
            out, scale, px, py, img_rgb.shape[1], img_rgb.shape[0]
        )
        return dets, infer_ms


def draw_detections(frame_bgr, detections):
    for cls_id, score, box in detections:
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, cls_id)} {score:.2f}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        y_text = max(th + 6, y1)
        cv2.rectangle(
            frame_bgr,
            (x1, y_text - th - 6),
            (x1 + tw + 6, y_text + 2),
            color,
            -1,
        )
        cv2.putText(
            frame_bgr,
            label,
            (x1 + 3, y_text - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def main():
    # Keep UI/camera on core 0 where possible.
    set_affinity([0])

    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (CAM_W, CAM_H), "format": "RGB888"},
        controls={"FrameRate": CAM_FPS},
        buffer_count=4,
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(1.0)

    frame_lock = threading.Lock()
    result_lock = threading.Lock()
    stop = threading.Event()

    latest = {"seq": -1, "frame": None}
    result = {
        "seq": -1,
        "detections": [],
        "infer_ms": 0.0,
        "error": None,
    }

    def worker():
        # Create NCNN and its internal threads after pinning this worker.
        set_affinity([1, 2, 3])

        try:
            detector = NCNNDetector()
        except Exception as e:
            with result_lock:
                result["error"] = f"Model init failed: {e}"
            stop.set()
            return

        seen = -1

        while not stop.is_set():
            with frame_lock:
                seq = latest["seq"]
                frm = latest["frame"]

            if frm is None or seq == seen:
                time.sleep(0.001)
                continue

            seen = seq

            try:
                dets, infer_ms = detector.infer(frm)
                with result_lock:
                    result["seq"] = seq
                    result["detections"] = dets
                    result["infer_ms"] = infer_ms
                    result["error"] = None
            except Exception as e:
                with result_lock:
                    result["error"] = str(e)
                stop.set()
                return

    thread = threading.Thread(target=worker, name="ncnn-worker", daemon=True)
    thread.start()

    seq = 0
    frames = 0
    t_fps = time.perf_counter()
    display_fps = 0.0

    try:
        while not stop.is_set():
            rgb = picam2.capture_array()

            with frame_lock:
                latest["seq"] = seq
                latest["frame"] = rgb
            seq += 1

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            with result_lock:
                dets = list(result["detections"])
                infer_ms = float(result["infer_ms"])
                err = result["error"]

            draw_detections(bgr, dets)

            frames += 1
            now = time.perf_counter()
            elapsed = now - t_fps
            if elapsed >= 1.0:
                display_fps = frames / elapsed
                frames = 0
                t_fps = now

            infer_fps = (1000.0 / infer_ms) if infer_ms > 0 else 0.0

            cv2.putText(
                bgr,
                f"Display {display_fps:.1f} FPS | NCNN {infer_fps:.1f} FPS | {infer_ms:.0f} ms",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if err:
                cv2.putText(
                    bgr,
                    err[:90],
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Traffic Sign NCNN - Pi 4", bgr)
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
