"""
ADAS REALTIME CAMERA - Raspberry Pi 4 (8GB)
NATIVE NCNN INFERENCE BUILD
OPTIMIZED DECODER BUILD

Architecture:
    Main process:
        Picamera2 -> ADAS overlay -> cv2.imshow
        Target: ~25 FPS

    Inference process:
        shared-memory latest-frame buffer -> native ncnn.Net
        YOLO11n-seg decode -> NMS -> lane masks + object boxes
        Target: fastest stable NCNN inference on the Pi

Important:
    This file intentionally DOES NOT import Ultralytics/YOLO for inference.
    The exported NCNN model is loaded directly with ncnn.Net().
    Lane classes keep segmentation masks; ordinary objects use boxes to avoid
    expensive full-resolution mask reconstruction for every detection.

Your measured native NCNN optimum:
    1 thread = 4.04 FPS
    2 threads = 6.64 FPS
    3 threads = 7.26 FPS  <- selected
    4 threads = 6.64 FPS

Keys:
    q = quit
    s = save current annotated frame
"""

import os

# Keep unrelated BLAS thread pools from stealing CPU from NCNN.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import shared_memory
from collections import deque
from queue import Empty


# ============================================================
# 0. MODEL / SPEED SETTINGS
# ============================================================

NCNN_DIR = "/home/soham/Dashcam/best_ncnn_model"

INFERENCE_SIZE = 320
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 20

# Exported YOLO11n-seg has 32 mask coefficients.
MASK_COEFFS = 32

CAMERA_W = 640
CAMERA_H = 480
CAMERA_FPS = 25

# Keep X11 rendering light.
SHOW_SCALE = 0.75

# Native NCNN benchmark proved 3 threads are fastest on this Pi.
NCNN_THREADS = 3

# Core assignment is deliberately different from the old 2-core/2-core split.
# NCNN needs 3 cores to reach its measured peak. The camera/display process
# gets core 0; the inference process gets cores 1-3.
MAIN_CPU_SET = {0}
INFER_CPU_SET = {1, 2, 3}


# ============================================================
# 1. CLASS MAP
# ============================================================

class_names = {
    0: "Bus",
    1: "Car",
    2: "Dashed Lane",
    3: "Pedestrian",
    4: "Truck",
    5: "Yellow Lane",
}

class_colors = {
    0: (0, 165, 255),   # Bus
    1: (0, 0, 255),     # Car
    2: (0, 255, 0),     # Dashed Lane
    3: (255, 0, 255),   # Pedestrian
    4: (255, 0, 0),     # Truck
    5: (0, 255, 255),   # Yellow Lane
}

LANE_CLASSES = (2, 5)


# ============================================================
# 2. PERIMETER ZONE
# ============================================================

raw_pts = [
    [440, 430],
    [548, 270],
    [710, 268],
    [828, 411],
    [712, 398],
    [632, 396],
    [553, 405],
    [440, 430],
]


def set_cpu_affinity(cpu_set, label):
    try:
        os.sched_setaffinity(0, cpu_set)
        print(f"{label} CPU affinity: {sorted(cpu_set)}")
    except (AttributeError, PermissionError, OSError) as e:
        print(f"{label} CPU affinity not applied: {e}")


def sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def ncnn_mat_to_array(mat, shape):
    """Convert an ncnn.Mat to a float32 NumPy array with the known shape."""
    arr = np.asarray(mat, dtype=np.float32)
    return arr.reshape(shape)


# ============================================================
# 3. LETTERBOX / PREPROCESS
# ============================================================

def letterbox(image, new_shape=(320, 320), color=(114, 114, 114)):
    """Ultralytics-style centered letterbox for 640x480 -> 320x320."""
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = new_shape

    scale = min(dst_w / float(src_w), dst_h / float(src_h))

    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    dw = dst_w - new_w
    dh = dst_h - new_h

    left = dw // 2
    right = dw - left
    top = dh // 2
    bottom = dh - top

    out = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return out, scale, left, top, new_w, new_h


def make_ncnn_input(frame_bgr):
    """Preprocess a camera BGR frame into an NCNN RGB input blob."""
    padded, scale, pad_x, pad_y, content_w, content_h = letterbox(
        frame_bgr,
        (INFERENCE_SIZE, INFERENCE_SIZE),
    )

    # Ultralytics feeds RGB to the exported YOLO model.
    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    mat = ncnn.Mat.from_pixels(
        padded_rgb,
        ncnn.Mat.PixelType.PIXEL_RGB,
        INFERENCE_SIZE,
        INFERENCE_SIZE,
    )

    # YOLO export expects normalized 0..1 image input.
    mat.substract_mean_normalize(
        [0.0, 0.0, 0.0],
        [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],
    )

    meta = (
        scale,
        pad_x,
        pad_y,
        content_w,
        content_h,
    )

    return mat, meta


# ============================================================
# 4. YOLO11 SEGMENTATION DECODER
# ============================================================

def classwise_nms(boxes_xywh, scores, class_ids):
    """Class-aware OpenCV NMS; returns indices into the candidate arrays."""
    if len(boxes_xywh) == 0:
        return []

    selected = []

    for cls in np.unique(class_ids):
        inds = np.where(class_ids == cls)[0]

        if len(inds) == 0:
            continue

        boxes = boxes_xywh[inds].tolist()
        cls_scores = scores[inds].tolist()

        kept = cv2.dnn.NMSBoxes(
            boxes,
            cls_scores,
            CONF_THRESHOLD,
            NMS_IOU_THRESHOLD,
        )

        if kept is None or len(kept) == 0:
            continue

        kept = np.asarray(kept).reshape(-1)
        selected.extend(inds[kept].tolist())

    if not selected:
        return []

    selected.sort(key=lambda i: float(scores[i]), reverse=True)
    return selected[:MAX_DETECTIONS]


def decode_yolo_segmentation_fast(
    out0,
    out1,
    original_shape,
    letterbox_meta,
):
    """
    Fast YOLO11-seg decoder for Raspberry Pi.

    Key optimization:
      - NMS is applied before any mask reconstruction.
      - Lane classes (2, 5) get masks because steering needs them.
      - Vehicle/person classes use their YOLO bounding boxes directly;
        we do not reconstruct a segmentation mask for every object.
      - Lane contours are generated on a compact 160x120 canvas instead
        of a full 640x480 mask.
      - A pre-NMS top-K cap keeps OpenCV NMS small.

    Exported NCNN outputs on this Pi:
      out0 = 42 x 2100
      out1 = 32 x 80 x 80
    """
    orig_h, orig_w = original_shape[:2]

    pred = ncnn_mat_to_array(out0, (int(out0.h), int(out0.w))).T
    proto = ncnn_mat_to_array(
        out1,
        (int(out1.c), int(out1.h), int(out1.w)),
    )

    if pred.shape[1] <= 4 + MASK_COEFFS:
        return []

    n_mask = int(proto.shape[0])
    num_classes = pred.shape[1] - 4 - n_mask
    if num_classes <= 0:
        return []

    xywh = pred[:, :4].astype(np.float32, copy=False)
    class_scores = pred[:, 4:4 + num_classes].astype(np.float32, copy=False)
    mask_coeff = pred[:, 4 + num_classes:4 + num_classes + n_mask]

    # Exported NCNN output contains class logits for this model.
    if class_scores.size:
        if float(class_scores.min()) < 0.0 or float(class_scores.max()) > 1.0:
            class_scores = sigmoid(class_scores)

    class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    keep = scores >= CONF_THRESHOLD
    if not np.any(keep):
        return []

    xywh = xywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    mask_coeff = mask_coeff[keep]

    # --------------------------------------------------------
    # Pre-NMS top-K: NMS does not need all 2100 low-score candidates.
    # --------------------------------------------------------
    PRE_NMS_TOPK = 300
    if len(scores) > PRE_NMS_TOPK:
        top = np.argpartition(scores, -PRE_NMS_TOPK)[-PRE_NMS_TOPK:]
        top = top[np.argsort(scores[top])[::-1]]
        xywh = xywh[top]
        scores = scores[top]
        class_ids = class_ids[top]
        mask_coeff = mask_coeff[top]

    # --------------------------------------------------------
    # xywh -> xyxy in 320x320 letterboxed coordinates.
    # --------------------------------------------------------
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] * 0.5
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] * 0.5
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] * 0.5
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] * 0.5

    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, INFERENCE_SIZE - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, INFERENCE_SIZE - 1)

    nms_boxes = np.empty_like(xywh)
    nms_boxes[:, 0] = boxes[:, 0]
    nms_boxes[:, 1] = boxes[:, 1]
    nms_boxes[:, 2] = np.maximum(0, boxes[:, 2] - boxes[:, 0])
    nms_boxes[:, 3] = np.maximum(0, boxes[:, 3] - boxes[:, 1])

    nms_keep = classwise_nms(nms_boxes, scores, class_ids)
    if not nms_keep:
        return []

    boxes = boxes[nms_keep]
    scores = scores[nms_keep]
    class_ids = class_ids[nms_keep]
    mask_coeff = mask_coeff[nms_keep]

    scale, pad_x, pad_y, content_w, content_h = letterbox_meta

    # Map letterboxed 320x320 boxes directly to camera coordinates.
    inv_scale = 1.0 / float(scale)
    boxes_orig = boxes.copy()
    boxes_orig[:, 0] = (boxes[:, 0] - pad_x) * inv_scale
    boxes_orig[:, 1] = (boxes[:, 1] - pad_y) * inv_scale
    boxes_orig[:, 2] = (boxes[:, 2] - pad_x) * inv_scale
    boxes_orig[:, 3] = (boxes[:, 3] - pad_y) * inv_scale
    boxes_orig[:, 0::2] = np.clip(boxes_orig[:, 0::2], 0, orig_w - 1)
    boxes_orig[:, 1::2] = np.clip(boxes_orig[:, 1::2], 0, orig_h - 1)

    payload = []

    # --------------------------------------------------------
    # Fast path for ordinary objects: use bounding boxes.
    # --------------------------------------------------------
    lane_indices = [
        i for i, cls in enumerate(class_ids)
        if int(cls) in LANE_CLASSES
    ]

    for i in range(len(nms_keep)):
        cls = int(class_ids[i])
        if cls in LANE_CLASSES:
            continue

        x1, y1, x2, y2 = boxes_orig[i].astype(np.int32)
        if x2 <= x1 or y2 <= y1:
            continue

        box_pts = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.int16,
        )
        payload.append((cls, box_pts, float(scores[i])))

    # --------------------------------------------------------
    # Lane-only mask reconstruction.
    # --------------------------------------------------------
    if lane_indices:
        proto_h, proto_w = int(proto.shape[1]), int(proto.shape[2])
        proto_flat = proto.reshape(n_mask, -1)
        lane_coeff = mask_coeff[lane_indices]

        # Only 1-2 lanes are normally present, so this matrix is tiny.
        lane_masks = sigmoid(lane_coeff @ proto_flat).reshape(
            len(lane_indices), proto_h, proto_w
        )

        # 640x480 camera corresponds to 4:3 content inside 320x320.
        # Use a compact 160x120 contour canvas.
        lane_out_w = 160
        lane_out_h = 120
        content_x1_p = int(round(pad_x * proto_w / INFERENCE_SIZE))
        content_x2_p = int(round((pad_x + content_w) * proto_w / INFERENCE_SIZE))
        content_y1_p = int(round(pad_y * proto_h / INFERENCE_SIZE))
        content_y2_p = int(round((pad_y + content_h) * proto_h / INFERENCE_SIZE))

        content_x1_p = max(0, min(proto_w, content_x1_p))
        content_x2_p = max(0, min(proto_w, content_x2_p))
        content_y1_p = max(0, min(proto_h, content_y1_p))
        content_y2_p = max(0, min(proto_h, content_y2_p))

        if content_x2_p > content_x1_p and content_y2_p > content_y1_p:
            for j, det_i in enumerate(lane_indices):
                mask = lane_masks[j]

                # Crop in prototype space using the lane bounding box.
                bx1 = int(np.floor(boxes[det_i, 0] * proto_w / INFERENCE_SIZE))
                by1 = int(np.floor(boxes[det_i, 1] * proto_h / INFERENCE_SIZE))
                bx2 = int(np.ceil(boxes[det_i, 2] * proto_w / INFERENCE_SIZE))
                by2 = int(np.ceil(boxes[det_i, 3] * proto_h / INFERENCE_SIZE))

                bx1 = max(0, min(proto_w - 1, bx1))
                by1 = max(0, min(proto_h - 1, by1))
                bx2 = max(0, min(proto_w, bx2))
                by2 = max(0, min(proto_h, by2))

                if bx2 <= bx1 or by2 <= by1:
                    continue

                cropped = np.zeros_like(mask, dtype=np.float32)
                cropped[by1:by2, bx1:bx2] = mask[by1:by2, bx1:bx2]

                content = cropped[
                    content_y1_p:content_y2_p,
                    content_x1_p:content_x2_p,
                ]

                if content.size == 0:
                    continue

                # Directly create a small 160x120 camera-space mask.
                small = cv2.resize(
                    content,
                    (lane_out_w, lane_out_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                binary = (small > 0.5).astype(np.uint8)

                if int(binary.sum()) < 2:
                    continue

                contours, _ = cv2.findContours(
                    binary,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if not contours:
                    continue

                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                if len(contour) < 3 or area < 1.0:
                    continue

                pts = contour.reshape(-1, 2).astype(np.float32)
                pts[:, 0] *= orig_w / float(lane_out_w)
                pts[:, 1] *= orig_h / float(lane_out_h)
                pts[:, 0] = np.clip(pts[:, 0], 0, orig_w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, orig_h - 1)

                payload.append(
                    (
                        int(class_ids[det_i]),
                        pts.astype(np.int16),
                        float(scores[det_i]),
                    )
                )

    return payload


# ============================================================
# 5. NATIVE NCNN INFERENCE PROCESS
# ============================================================

def inference_process(
    shm_name,
    frame_shape,
    frame_lock,
    frame_seq,
    result_queue,
    stop_event,
    ready_event,
    error_queue,
):
    shm = None

    try:
        set_cpu_affinity(INFER_CPU_SET, "Inference process")

        global ncnn
        import ncnn

        cv2.setNumThreads(1)

        if not os.path.isdir(NCNN_DIR):
            raise FileNotFoundError(
                f"NCNN model folder not found: {NCNN_DIR}"
            )

        param_path = os.path.join(NCNN_DIR, "model.ncnn.param")
        bin_path = os.path.join(NCNN_DIR, "model.ncnn.bin")

        if not os.path.exists(param_path):
            raise FileNotFoundError(param_path)
        if not os.path.exists(bin_path):
            raise FileNotFoundError(bin_path)

        height, width, channels = frame_shape

        shm = shared_memory.SharedMemory(name=shm_name)
        frame_view = np.ndarray(
            frame_shape,
            dtype=np.uint8,
            buffer=shm.buf,
        )

        print("============================================================")
        print("NATIVE NCNN WORKER")
        print("============================================================")
        print(f"Model       : {NCNN_DIR}")
        print(f"NCNN threads: {NCNN_THREADS}")
        print("Vulkan      : False")
        print("Input       : 320x320")
        print("Backend     : native ncnn.Net()")
        print("Decoder     : lane masks + object boxes")
        print("============================================================")

        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        net.opt.num_threads = NCNN_THREADS

        if hasattr(net.opt, "use_winograd_convolution"):
            net.opt.use_winograd_convolution = True

        print("Loading NCNN model...")

        ret = net.load_param(param_path)
        if ret != 0:
            raise RuntimeError(f"load_param failed: {ret}")

        ret = net.load_model(bin_path)
        if ret != 0:
            raise RuntimeError(f"load_model failed: {ret}")

        print("NCNN model loaded")
        print("Native NCNN warmup...")

        dummy = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )
        dummy_mat, dummy_meta = make_ncnn_input(dummy)

        for _ in range(3):
            ex = net.create_extractor()
            ex.input("in0", dummy_mat)
            r0, o0 = ex.extract("out0")
            r1, o1 = ex.extract("out1")
            if r0 != 0 or r1 != 0:
                raise RuntimeError(
                    f"Warmup extraction failed: out0={r0}, out1={r1}"
                )

        print(
            f"NCNN warmup done | outputs: "
            f"out0={o0.h}x{o0.w}, "
            f"out1={o1.c}x{o1.h}x{o1.w}"
        )

        ready_event.set()

        last_sequence = -1
        fps_samples = deque(maxlen=30)

        while not stop_event.is_set():
            if frame_seq.value == last_sequence:
                time.sleep(0.001)
                continue

            with frame_lock:
                current_sequence = int(frame_seq.value)
                local_frame = frame_view.copy()

            if current_sequence == last_sequence:
                continue

            last_sequence = current_sequence

            total_start = time.perf_counter()
            pre_ms = engine_ms = decode_ms = 0.0

            try:
                t0 = time.perf_counter()
                input_mat, letterbox_meta = make_ncnn_input(local_frame)
                pre_ms = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                ex = net.create_extractor()
                ex.input("in0", input_mat)

                ret0, out0 = ex.extract("out0")
                ret1, out1 = ex.extract("out1")

                if ret0 != 0 or ret1 != 0:
                    raise RuntimeError(
                        f"NCNN extraction failed: out0={ret0}, out1={ret1}"
                    )
                engine_ms = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                payload = decode_yolo_segmentation_fast(
                    out0,
                    out1,
                    local_frame.shape,
                    letterbox_meta,
                )
                decode_ms = (time.perf_counter() - t0) * 1000.0

            except Exception as e:
                print(f"Native NCNN inference error: {e}")
                payload = []

            total_ms = (time.perf_counter() - total_start) * 1000.0
            fps = 1000.0 / total_ms if total_ms > 0 else 0.0
            fps_samples.append(fps)
            avg_fps = float(np.mean(fps_samples)) if fps_samples else fps

            # Only keep latest result.
            try:
                while True:
                    result_queue.get_nowait()
            except Empty:
                pass

            try:
                result_queue.put_nowait(
                    (payload, avg_fps, current_sequence, pre_ms, engine_ms, decode_ms, total_ms)
                )
            except Exception:
                pass

        shm.close()

    except Exception as e:
        try:
            error_queue.put_nowait(str(e))
        except Exception:
            pass
        ready_event.set()

    finally:
        ready_event.set()
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


# ============================================================
# 6. MAIN PROCESS
# ============================================================

def main():
    set_cpu_affinity(MAIN_CPU_SET, "Main process")
    cv2.setNumThreads(1)

    from picamera2 import Picamera2

    if not os.path.isdir(NCNN_DIR):
        raise FileNotFoundError(
            f"NCNN model folder not found: {NCNN_DIR}"
        )

    w, h = CAMERA_W, CAMERA_H

    sx = w / 1280.0
    sy = h / 720.0

    zone_points = np.array(
        [[int(x * sx), int(y * sy)] for x, y in raw_pts],
        dtype=np.int32,
    )

    poly_center_x = (
        int(zone_points[:, 0].min())
        + int(zone_points[:, 0].max())
    ) // 2

    # Precomputed containment lookup.
    zone_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(zone_mask, [zone_points], 1)

    radar_w, radar_h = 130, 210
    radar_scale_x = radar_w / float(w)
    radar_scale_y = radar_h / float(h)

    # --------------------------------------------------------
    # Shared memory: exactly one newest frame.
    # --------------------------------------------------------
    shm_size = h * w * 3
    shm = shared_memory.SharedMemory(
        create=True,
        size=shm_size,
    )

    shared_frame = np.ndarray(
        (h, w, 3),
        dtype=np.uint8,
        buffer=shm.buf,
    )

    ctx = mp.get_context("spawn")

    frame_lock = ctx.Lock()
    frame_seq = ctx.Value("q", 0)
    stop_event = ctx.Event()
    ready_event = ctx.Event()

    result_queue = ctx.Queue(maxsize=1)
    error_queue = ctx.Queue(maxsize=2)

    worker = ctx.Process(
        target=inference_process,
        args=(
            shm.name,
            (h, w, 3),
            frame_lock,
            frame_seq,
            result_queue,
            stop_event,
            ready_event,
            error_queue,
        ),
        daemon=True,
    )

    worker.start()

    print()
    print("============================================================")
    print("ADAS NATIVE-NCNN 25-FPS CAMERA/DISPLAY MODE")
    print("============================================================")
    print(f"Backend       : NATIVE NCNN")
    print(f"Model         : {NCNN_DIR}")
    print(f"NCNN threads  : {NCNN_THREADS}")
    print(f"YOLO input    : {INFERENCE_SIZE}x{INFERENCE_SIZE}")
    print(f"Camera        : {w}x{h}@{CAMERA_FPS} FPS")
    print("Architecture  : MAIN PROCESS + NATIVE-NCNN PROCESS")
    print("Frame buffer  : shared memory, latest frame only")
    print("Inference     : direct ncnn.Net(), no Ultralytics")
    print("Decoder       : lane masks + object boxes (optimized)")
    print("============================================================")
    print("Waiting for native NCNN warmup...")

    if not ready_event.wait(timeout=90):
        raise RuntimeError("Native NCNN worker warmup timed out")

    try:
        err = error_queue.get_nowait()
    except Empty:
        err = None

    if err:
        stop_event.set()
        worker.join(timeout=2)
        raise RuntimeError(f"Native NCNN worker failed: {err}")

    print("Native NCNN worker online")
    print("Press 'q' to quit, 's' to save frame")
    print()

    picam2 = Picamera2()

    picam2.configure(
        picam2.create_video_configuration(
            main={
                "size": (w, h),
                "format": "RGB888",
            },
            controls={
                "FrameRate": CAMERA_FPS,
            },
            buffer_count=4,
        )
    )

    picam2.start()
    time.sleep(1.0)

    frame_index = 0
    display_times = deque(maxlen=60)
    yolo_times = deque(maxlen=30)

    last_display_t = time.perf_counter()
    last_result_seq = -1
    latest_payload = []
    pre_ms = engine_ms = decode_ms = total_ms = 0.0

    signal = "Path Clear"
    signal_color = (255, 255, 255)

    try:
        while not stop_event.is_set():
            # ------------------------------------------------
            # 1. Capture.
            # ------------------------------------------------
            frame_rgb = picam2.capture_array("main")
            frame = cv2.cvtColor(
                frame_rgb,
                cv2.COLOR_RGB2BGR,
            )

            frame_index += 1

            # ------------------------------------------------
            # 2. Publish newest frame.
            # ------------------------------------------------
            with frame_lock:
                shared_frame[:] = frame
                frame_seq.value += 1

            # ------------------------------------------------
            # 3. Consume latest inference result.
            # ------------------------------------------------
            try:
                while True:
                    payload, yolo_fps, result_seq, pre_ms, engine_ms, decode_ms, total_ms = result_queue.get_nowait()
                    latest_payload = payload
                    last_result_seq = int(result_seq)
                    yolo_times.append(float(yolo_fps))
            except Empty:
                pass

            # ------------------------------------------------
            # 4. Main overlay.
            # ------------------------------------------------
            overlay = frame.copy()

            zone_fill = overlay.copy()
            cv2.fillPoly(zone_fill, [zone_points], (100, 200, 255))
            cv2.polylines(
                overlay,
                [zone_points],
                True,
                (255, 255, 0),
                2,
            )
            overlay = cv2.addWeighted(
                zone_fill,
                0.25,
                overlay,
                0.75,
                0,
            )

            radar = np.zeros(
                (radar_h, radar_w, 3),
                dtype=np.uint8,
            )

            current_signal = "Path Clear"
            current_signal_color = (255, 255, 255)

            for cls, pts_raw, score in latest_payload:
                pts = np.asarray(pts_raw, dtype=np.int32)

                if len(pts) < 3:
                    continue

                cls = int(cls)
                color = class_colors.get(cls, (200, 200, 200))
                is_lane = cls in LANE_CLASSES

                if is_lane:
                    cv2.polylines(
                        overlay,
                        [pts],
                        False,
                        color,
                        3,
                    )
                else:
                    cv2.fillPoly(
                        overlay,
                        [pts],
                        color,
                    )

                # Radar.
                rp = pts.astype(np.float32)
                rp[:, 0] *= radar_scale_x
                rp[:, 1] *= radar_scale_y
                rp = rp.astype(np.int32)

                if is_lane:
                    cv2.polylines(
                        radar,
                        [rp],
                        False,
                        color,
                        2,
                    )
                else:
                    cv2.fillPoly(
                        radar,
                        [rp],
                        color,
                    )

                # Steering using precomputed zone mask.
                if is_lane:
                    sample_step = max(1, len(pts) // 50)
                    sample = pts[::sample_step]

                    xs = np.clip(sample[:, 0], 0, w - 1)
                    ys = np.clip(sample[:, 1], 0, h - 1)

                    inside = zone_mask[ys, xs]

                    if np.any(inside):
                        cx = float(np.mean(pts[:, 0]))

                        if cx < poly_center_x:
                            current_signal = "Turn Right"
                            current_signal_color = (0, 165, 255)
                        else:
                            current_signal = "Turn Left"
                            current_signal_color = (0, 255, 255)

                # Label.
                m = cv2.moments(pts)
                if m["m00"] > 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])

                    label = f"{class_names.get(cls, 'Obj')} {score:.2f}"

                    cv2.putText(
                        overlay,
                        label,
                        (max(0, cx - 25), max(15, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            signal = current_signal
            signal_color = current_signal_color

            output = cv2.addWeighted(
                overlay,
                0.6,
                frame,
                0.4,
                0,
            )

            # ------------------------------------------------
            # 5. Radar widget.
            # ------------------------------------------------
            tri = np.array(
                [
                    [radar_w // 2, radar_h - 15],
                    [radar_w // 2 - 12, radar_h - 35],
                    [radar_w // 2 + 12, radar_h - 35],
                ],
                dtype=np.int32,
            )

            cv2.drawContours(
                radar,
                [tri],
                0,
                (255, 0, 0),
                -1,
            )

            box_x = 16
            box_y = 16

            cv2.rectangle(
                output,
                (box_x, box_y),
                (
                    box_x + radar_w + 8,
                    box_y + radar_h + 8,
                ),
                (255, 255, 255),
                2,
            )

            output[
                box_y + 4:box_y + 4 + radar_h,
                box_x + 4:box_x + 4 + radar_w,
            ] = radar

            # ------------------------------------------------
            # 6. Status box.
            # ------------------------------------------------
            st_w, st_h = 280, 50
            st_x = w - st_w - 20
            st_y = 20

            roi = output[st_y:st_y + st_h, st_x:st_x + st_w]
            output[st_y:st_y + st_h, st_x:st_x + st_w] = cv2.addWeighted(
                roi,
                0.6,
                np.zeros_like(roi),
                0.4,
                0,
            )

            cv2.rectangle(
                output,
                (st_x - 3, st_y - 3),
                (st_x + st_w + 3, st_y + st_h + 3),
                (0, 255, 255),
                2,
            )

            status_text = f"STATUS: {signal}"
            ts = cv2.getTextSize(
                status_text,
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                2,
            )[0]

            cv2.putText(
                output,
                status_text,
                (
                    st_x + (st_w - ts[0]) // 2,
                    st_y + (st_h + ts[1]) // 2,
                ),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                signal_color,
                2,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # 7. FPS / result age.
            # ------------------------------------------------
            now = time.perf_counter()
            dt = now - last_display_t
            last_display_t = now

            if dt > 0:
                display_times.append(1.0 / dt)

            display_fps = (
                float(np.mean(display_times))
                if display_times
                else 0.0
            )

            yolo_avg = (
                float(np.mean(yolo_times))
                if yolo_times
                else 0.0
            )

            # How many camera frames have arrived since the newest result.
            result_age = max(0, int(frame_seq.value) - last_result_seq)

            cv2.putText(
                output,
                f"Disp FPS: {display_fps:.1f}",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                output,
                f"Total: {yolo_avg:.1f} FPS | Eng {engine_ms:.0f}ms | Dec {decode_ms:.0f}ms",
                (10, h - 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                output,
                f"Result age: {result_age}",
                (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            show = (
                cv2.resize(
                    output,
                    None,
                    fx=SHOW_SCALE,
                    fy=SHOW_SCALE,
                    interpolation=cv2.INTER_AREA,
                )
                if SHOW_SCALE != 1.0
                else output
            )

            cv2.imshow("ADAS - Raspberry Pi", show)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                filename = f"frame_{frame_index}.jpg"
                cv2.imwrite(filename, output)
                print(f"Saved: {filename}")

            if frame_index % 100 == 0:
                print(
                    f"Frames: {frame_index} | "
                    f"Disp FPS: {display_fps:.1f} | "
                    f"Total FPS: {yolo_avg:.1f} | "
                    f"Engine: {engine_ms:.1f}ms | "
                    f"Decode: {decode_ms:.1f}ms | "
                    f"Result age: {result_age} | "
                    f"Signal: {signal}"
                )

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        stop_event.set()

        try:
            picam2.stop()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        worker.join(timeout=3)

        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1)

        try:
            shm.close()
        finally:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

        avg_disp = (
            float(np.mean(display_times))
            if display_times
            else 0.0
        )

        avg_yolo = (
            float(np.mean(yolo_times))
            if yolo_times
            else 0.0
        )

        print()
        print("============================================================")
        print("SESSION COMPLETE")
        print(f"Frames       : {frame_index}")
        print(f"Display FPS  : {avg_disp:.1f}")
        print(f"Total FPS    : {avg_yolo:.1f}")
        print(f"Last engine  : {engine_ms:.1f} ms")
        print(f"Last decode  : {decode_ms:.1f} ms")
        print(f"NCNN threads : {NCNN_THREADS}")
        print("============================================================")


if __name__ == "__main__":
    mp.freeze_support()
    main()
