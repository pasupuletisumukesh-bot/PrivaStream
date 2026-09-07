import cv2
import os
import time
import threading
import urllib.request
import numpy as np
import pyvirtualcam
import onnxruntime as ort

from dataclasses import dataclass, field
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from insightface.model_zoo import get_model


# ============================================================
# PrivaStream - SAMPLE 10
# YOLO face detection + ArcFace recognition
# 30-second delayed privacy stream + 60 FPS output
# ============================================================

# ---------------- CONFIG ----------------

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TARGET_FPS = 60.0

BUFFER_DELAY_SECONDS = 30.0
SEGMENT_SECONDS = 2.0
BUFFER_CODEC = "MJPG"

YOLO_MODEL_PATH = "models/yolov11n-face.pt"
YOLO_MODEL_URL = (
    "https://github.com/YapaLab/yolo-face/releases/download/"
    "1.0.0/yolov11n-face.pt"
)
YOLO_IMGSZ = 640
YOLO_CONF = 0.35
YOLO_MAX_DET = 20

ARCFACE_MODEL = os.path.expanduser(
    "~/.insightface/models/buffalo_s/w600k_mbf.onnx"
)

# Keep these close to the values that worked in the previous samples.
SIMILARITY_THRESHOLD = 0.48
REID_THRESHOLD = 0.48

MAX_MATCH_DISTANCE = 180.0
MAX_LOST_FRAMES = 45
MAX_PREDICT_FRAMES = 12
VELOCITY_SMOOTHING = 0.35
PRIMARY_REVERIFY_TIMEOUT = 2.0

CAMERA_BUFFER_SIZE = 1
MAX_OUTPUT_QUEUE = 180


# ---------------- DATA ----------------

@dataclass
class FaceDetection:
    bbox: np.ndarray
    embedding: np.ndarray
    whitelisted: bool
    similarity: float
    confidence: float


@dataclass
class AIResult:
    seq_id: int = 0
    timestamp: float = 0.0
    latency_ms: float = 0.0
    fps: float = 0.0
    detections: List[FaceDetection] = field(default_factory=list)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    embedding: np.ndarray
    velocity: np.ndarray
    whitelisted: bool = False
    confidence: float = 0.0
    lost_frames: int = 0
    predict_frames: int = 0
    last_verified_time: float = 0.0
    manual_revealed: bool = False


@dataclass
class BufferedSegment:
    sequence: int
    path: str
    start_time: float
    end_time: float


# ---------------- HELPERS ----------------

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / denom)


def center(bbox):
    return np.array(
        [(bbox[0] + bbox[2]) * 0.5,
         (bbox[1] + bbox[3]) * 0.5],
        dtype=np.float32
    )


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def clamp_bbox(bbox, width, height):
    return np.array([
        max(0, min(width - 1, int(bbox[0]))),
        max(0, min(height - 1, int(bbox[1]))),
        max(0, min(width, int(bbox[2]))),
        max(0, min(height, int(bbox[3])))
    ], dtype=np.float32)


def crop_face(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    fw, fh = max(1, x2 - x1), max(1, y2 - y1)

    px, py = int(fw * 0.15), int(fh * 0.15)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# ---------------- MODEL SETUP ----------------

def load_yolo():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(YOLO_MODEL_PATH):
        print("[YOLO] Downloading yolov11n-face.pt...")
        urllib.request.urlretrieve(
            YOLO_MODEL_URL,
            YOLO_MODEL_PATH
        )

    try:
        import torch
        cuda = torch.cuda.is_available()
        device = 0 if cuda else "cpu"
        if cuda:
            print("[YOLO] CUDA:", torch.cuda.get_device_name(0))
        else:
            print("[YOLO] WARNING: PyTorch CUDA unavailable.")
    except Exception:
        device = "cpu"
        print("[YOLO] WARNING: PyTorch CUDA unavailable.")

    print("[YOLO] Loading:", YOLO_MODEL_PATH)
    model = YOLO(YOLO_MODEL_PATH)

    # GPU warm-up.
    dummy = np.zeros(
        (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
        dtype=np.uint8
    )
    model.predict(
        dummy,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        max_det=YOLO_MAX_DET,
        device=device,
        verbose=False
    )

    print("[YOLO] Ready.")
    return model, device


def load_arcface():
    if not os.path.exists(ARCFACE_MODEL):
        raise FileNotFoundError(
            f"ArcFace model not found: {ARCFACE_MODEL}"
        )

    providers = []
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    model = get_model(
        ARCFACE_MODEL,
        providers=providers
    )
    model.prepare(ctx_id=0)

    active = "UNKNOWN"
    try:
        active = model.session.get_providers()[0]
    except Exception:
        pass

    print("[ARCFACE] Provider:", active)
    return model, active


def get_embedding(arcface, frame, bbox):
    crop = crop_face(frame, bbox)
    if crop is None:
        return None

    crop = cv2.resize(
        crop,
        (112, 112),
        interpolation=cv2.INTER_LINEAR
    )

    try:
        feat = arcface.get_feat([crop])[0]
    except Exception:
        return None

    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feat)
    if norm > 1e-8:
        feat /= norm
    return feat


# ---------------- ASYNC AI ----------------

class AsyncAIWorker:
    """
    YOLO finds faces.
    ArcFace recognizes them.
    The worker always processes the newest available frame,
    so the video processor is never blocked by AI.
    """

    def __init__(self, yolo, yolo_device, arcface):
        self.yolo = yolo
        self.yolo_device = yolo_device
        self.arcface = arcface

        self.running = False
        self.thread = None

        self.frame_lock = threading.Lock()
        self.result_lock = threading.Lock()

        self.latest_frame = None
        self.latest_result = AIResult()

        self.whitelist = []
        self.sequence = 0

        self.fps = 0.0
        self.latency_ms = 0.0

        self.count = 0
        self.timer = time.perf_counter()

    def start(self):
        self.running = True
        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="PrivaStream-AI"
        )
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def submit(self, frame):
        with self.frame_lock:
            self.latest_frame = frame.copy()

    def get_result(self):
        with self.result_lock:
            return self.latest_result

    def set_whitelist(self, embeddings):
        with self.result_lock:
            self.whitelist = [x.copy() for x in embeddings]

    def _recognize(self, embedding):
        with self.result_lock:
            saved = list(self.whitelist)

        if not saved:
            return False, 0.0

        best = max(
            cosine_similarity(embedding, x)
            for x in saved
        )
        return best >= SIMILARITY_THRESHOLD, best

    def _loop(self):
        while self.running:
            with self.frame_lock:
                frame = self.latest_frame
                self.latest_frame = None

            if frame is None:
                time.sleep(0.001)
                continue

            start = time.perf_counter()
            detections = []

            try:
                results = self.yolo.predict(
                    frame,
                    imgsz=YOLO_IMGSZ,
                    conf=YOLO_CONF,
                    max_det=YOLO_MAX_DET,
                    device=self.yolo_device,
                    verbose=False
                )

                boxes = results[0].boxes

                if boxes is not None:
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].detach().cpu().numpy()
                        conf = float(
                            boxes.conf[i].detach().cpu().item()
                        )

                        bbox = clamp_bbox(
                            bbox,
                            frame.shape[1],
                            frame.shape[0]
                        )

                        emb = get_embedding(
                            self.arcface,
                            frame,
                            bbox
                        )
                        if emb is None:
                            continue

                        primary, similarity = self._recognize(emb)

                        detections.append(
                            FaceDetection(
                                bbox=bbox,
                                embedding=emb,
                                whitelisted=primary,
                                similarity=similarity,
                                confidence=conf
                            )
                        )

            except Exception as e:
                print("[AI] Error:", e)
                continue

            latency = (
                time.perf_counter() - start
            ) * 1000.0

            with self.result_lock:
                self.sequence += 1
                self.latest_result = AIResult(
                    seq_id=self.sequence,
                    timestamp=time.time(),
                    latency_ms=latency,
                    fps=self.fps,
                    detections=detections
                )
                self.latency_ms = latency

            self.count += 1
            elapsed = time.perf_counter() - self.timer

            if elapsed >= 1.0:
                self.fps = self.count / elapsed
                self.count = 0
                self.timer = time.perf_counter()


# ---------------- TRACKER ----------------

class HybridTracker:
    def __init__(self):
        self.active_tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.last_ai_seq = -1

    def reset(self):
        self.active_tracks.clear()
        self.next_id = 0
        self.last_ai_seq = -1

    def step(self, result, now):
        if result.seq_id > self.last_ai_seq:
            self._apply(result, now)
            self.last_ai_seq = result.seq_id
        else:
            self._predict(now)

    def _apply(self, result, now):
        detections = result.detections

        for track in self.active_tracks.values():
            track.bbox += track.velocity
            track.lost_frames += 1
            track.predict_frames += 1

        candidates = []

        for tid, track in self.active_tracks.items():
            for di, det in enumerate(detections):
                dist = float(
                    np.linalg.norm(
                        center(track.bbox) - center(det.bbox)
                    )
                )
                overlap = iou(track.bbox, det.bbox)
                emb_sim = cosine_similarity(
                    track.embedding,
                    det.embedding
                )

                if (
                    dist <= MAX_MATCH_DISTANCE
                    or overlap >= 0.05
                    or emb_sim >= REID_THRESHOLD
                ):
                    score = dist - overlap * 300.0 - emb_sim * 120.0
                    candidates.append((score, tid, di))

        candidates.sort(key=lambda x: x[0])

        matched_tracks = set()
        matched_detections = set()

        for _, tid, di in candidates:
            if tid in matched_tracks or di in matched_detections:
                continue

            track = self.active_tracks[tid]
            det = detections[di]

            old_bbox = track.bbox.copy()
            track.bbox = det.bbox.copy()

            movement = track.bbox - old_bbox
            track.velocity = (
                VELOCITY_SMOOTHING * track.velocity
                + (1.0 - VELOCITY_SMOOTHING) * movement
            )

            track.embedding = det.embedding.copy()
            track.confidence = det.similarity
            track.lost_frames = 0
            track.predict_frames = 0

            if det.whitelisted:
                track.whitelisted = True
                track.last_verified_time = now
            elif (
                now - track.last_verified_time
                > PRIMARY_REVERIFY_TIMEOUT
            ):
                track.whitelisted = False

            matched_tracks.add(tid)
            matched_detections.add(di)

        # New faces.
        for di, det in enumerate(detections):
            if di in matched_detections:
                continue

            self.active_tracks[self.next_id] = Track(
                track_id=self.next_id,
                bbox=det.bbox.copy(),
                embedding=det.embedding.copy(),
                velocity=np.zeros(4, dtype=np.float32),
                whitelisted=det.whitelisted,
                confidence=det.similarity,
                last_verified_time=(
                    now if det.whitelisted else 0.0
                )
            )
            self.next_id += 1

        # Delete lost tracks.
        dead = [
            tid for tid, track in self.active_tracks.items()
            if track.lost_frames > MAX_LOST_FRAMES
        ]
        for tid in dead:
            del self.active_tracks[tid]

    def _predict(self, now):
        dead = []

        for tid, track in self.active_tracks.items():
            track.lost_frames += 1
            track.predict_frames += 1

            if (
                track.whitelisted
                and
                now - track.last_verified_time
                > PRIMARY_REVERIFY_TIMEOUT
            ):
                track.whitelisted = False

            if track.predict_frames <= MAX_PREDICT_FRAMES:
                track.bbox += track.velocity
                track.velocity *= 0.98
            else:
                track.velocity *= 0.5

            if track.lost_frames > MAX_LOST_FRAMES:
                dead.append(tid)

        for tid in dead:
            del self.active_tracks[tid]


# ---------------- PRIVACY / CLICK STATE ----------------

class PrivacyManager:
    def __init__(self):
        self.primary_embeddings = []
        self.manual_reveals = []
        self.live_click = None
        self.output_click = None
        self.lock = threading.Lock()

    def live_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.lock:
                self.live_click = (x, y)

    def output_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.lock:
                self.output_click = (x, y)

    def consume_live_click(self):
        with self.lock:
            x = self.live_click
            self.live_click = None
            return x

    def consume_output_click(self):
        with self.lock:
            x = self.output_click
            self.output_click = None
            return x

    def reset(self, tracker, ai):
        self.primary_embeddings.clear()
        self.manual_reveals.clear()
        tracker.reset()
        ai.set_whitelist([])
        print("[PRIVACY] Primary/reveals reset.")

    def enroll_primary(self, result, click, ai):
        if click is None:
            return

        cx, cy = click
        best = None
        best_dist = float("inf")

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                d = float(
                    np.linalg.norm(
                        center(det.bbox)
                        - np.array([cx, cy], dtype=np.float32)
                    )
                )
                if d < best_dist:
                    best_dist = d
                    best = det

        if best is None:
            print("[PRIVACY] No AI face at live click.")
            return

        self.primary_embeddings = [best.embedding.copy()]
        ai.set_whitelist(self.primary_embeddings)

        print("[PRIVACY] PRIMARY enrolled.")

    def toggle_track(self, track):
        if track.whitelisted:
            print("[PRIVACY] Primary cannot be manually masked.")
            return

        best_i = None
        best_sim = -1.0

        for i, saved in enumerate(self.manual_reveals):
            sim = cosine_similarity(track.embedding, saved)
            if sim > best_sim:
                best_i = i
                best_sim = sim

        revealed = (
            track.manual_revealed
            or (
                best_i is not None
                and best_sim >= REID_THRESHOLD
            )
        )

        if revealed:
            track.manual_revealed = False
            if best_i is not None and best_sim >= REID_THRESHOLD:
                self.manual_reveals.pop(best_i)
            print("[PRIVACY] Bystander MASKED.")
        else:
            track.manual_revealed = True
            self.manual_reveals.append(track.embedding.copy())
            print("[PRIVACY] Bystander REVEALED.")

    def persistent_reveal(self, track):
        if track.manual_revealed:
            return True

        return any(
            cosine_similarity(track.embedding, saved)
            >= REID_THRESHOLD
            for saved in self.manual_reveals
        )


def clicked_track(tracker, click):
    if click is None:
        return None

    cx, cy = click
    best = None
    best_dist = float("inf")

    for track in tracker.active_tracks.values():
        x1, y1, x2, y2 = track.bbox

        if x1 <= cx <= x2 and y1 <= cy <= y2:
            d = float(
                np.linalg.norm(
                    center(track.bbox)
                    - np.array([cx, cy], dtype=np.float32)
                )
            )
            if d < best_dist:
                best_dist = d
                best = track

    return best


# ---------------- EMOJI ----------------

EMOJI_TEMPLATE = None


def init_emoji():
    global EMOJI_TEMPLATE

    font_path = r"C:\Windows\Fonts\seguiemj.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError(
            "C:\\Windows\\Fonts\\seguiemj.ttf not found."
        )

    image = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 100)

    draw.text(
        (64, 64),
        "😎",
        font=font,
        anchor="mm",
        embedded_color=True
    )

    EMOJI_TEMPLATE = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGBA2BGRA
    )


def apply_emoji(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)

    fw, fh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(fw * 0.10), int(fh * 0.10)

    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)

    tw, th = x2 - x1, y2 - y1
    if tw <= 0 or th <= 0:
        return

    scale = min(
        tw / EMOJI_TEMPLATE.shape[1],
        th / EMOJI_TEMPLATE.shape[0]
    )
    ew = max(1, int(EMOJI_TEMPLATE.shape[1] * scale))
    eh = max(1, int(EMOJI_TEMPLATE.shape[0] * scale))

    emoji = cv2.resize(
        EMOJI_TEMPLATE,
        (ew, eh),
        interpolation=cv2.INTER_AREA
    )

    px = x1 + (tw - ew) // 2
    py = y1 + (th - eh) // 2
    px2, py2 = min(w, px + ew), min(h, py + eh)

    if px >= px2 or py >= py2:
        return

    rgb = emoji[:py2-py, :px2-px, :3]
    alpha = (
        emoji[:py2-py, :px2-px, 3].astype(np.float32) / 255.0
    )[..., None]

    roi = frame[py:py2, px:px2]
    frame[py:py2, px:px2] = np.clip(
        roi.astype(np.float32) * (1.0 - alpha)
        + rgb.astype(np.float32) * alpha,
        0,
        255
    ).astype(np.uint8)


# ---------------- CAMERA BUFFER ----------------

class TemporalCameraBuffer:
    def __init__(self, directory):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

        self.lock = threading.Lock()
        self.segments = []
        self.latest_live_frame = None

        self.width = 0
        self.height = 0
        self.fps = TARGET_FPS

        self.running = False
        self.error = None
        self.thread = None

    def start(self, camera_index=0):
        self.running = True
        self.thread = threading.Thread(
            target=self._loop,
            args=(camera_index,),
            daemon=True
        )
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def get_live_frame(self):
        with self.lock:
            return (
                None
                if self.latest_live_frame is None
                else self.latest_live_frame.copy()
            )

    def get_ready_segments(self):
        cutoff = time.time() - BUFFER_DELAY_SECONDS
        with self.lock:
            return sorted(
                [
                    x for x in self.segments
                    if x.end_time <= cutoff
                ],
                key=lambda x: x.sequence
            )

    def remove_segment(self, sequence):
        path = None
        with self.lock:
            keep = []
            for s in self.segments:
                if s.sequence == sequence:
                    path = s.path
                else:
                    keep.append(s)
            self.segments = keep

        if path:
            try:
                os.remove(path)
            except OSError:
                pass

    def buffered_seconds(self):
        with self.lock:
            if not self.segments:
                return 0.0
            return max(
                0.0,
                self.segments[-1].end_time
                - self.segments[0].start_time
            )

    def _loop(self, camera_index):
        cap = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW
        )

        if not cap.isOpened():
            self.error = "Could not open camera."
            self.running = False
            return

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            CAMERA_BUFFER_SIZE
        )
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )
        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )
        cap.set(
            cv2.CAP_PROP_FPS,
            TARGET_FPS
        )

        self.width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        self.height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        self.fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS

        frames_per_segment = max(
            1,
            int(self.fps * SEGMENT_SECONDS)
        )

        fourcc = cv2.VideoWriter_fourcc(*BUFFER_CODEC)

        writer = None
        sequence = 0
        frame_count = 0
        start_time = 0.0
        path = None

        print(
            f"[CAMERA] {self.width}x{self.height} @ {self.fps:.1f} FPS"
        )
        print(
            f"[BUFFER] {BUFFER_DELAY_SECONDS:.0f}s disk-backed buffer"
        )

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.error = "Camera frame capture failed."
                    break

                now = time.time()

                with self.lock:
                    self.latest_live_frame = frame.copy()

                if writer is None:
                    path = os.path.join(
                        self.directory,
                        f"segment_{sequence:08d}.avi"
                    )
                    writer = cv2.VideoWriter(
                        path,
                        fourcc,
                        self.fps,
                        (self.width, self.height)
                    )

                    if not writer.isOpened():
                        self.error = "Could not create buffer file."
                        break

                    frame_count = 0
                    start_time = now

                writer.write(frame)
                frame_count += 1

                if frame_count >= frames_per_segment:
                    writer.release()
                    writer = None

                    with self.lock:
                        self.segments.append(
                            BufferedSegment(
                                sequence,
                                path,
                                start_time,
                                now
                            )
                        )

                    sequence += 1

        finally:
            if writer is not None:
                writer.release()
            cap.release()
            self.running = False


# ---------------- DELAYED PROCESSOR ----------------

class StreamProcessor:
    def __init__(self, camera_buffer, ai, tracker, privacy):
        self.camera_buffer = camera_buffer
        self.ai = ai
        self.tracker = tracker
        self.privacy = privacy

        self.running = False
        self.thread = None

        self.condition = threading.Condition()
        self.output_queue = []

        self.processing_fps = 0.0
        self.processed_count = 0
        self.timer = time.perf_counter()
        self.count = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(
            target=self._loop,
            daemon=True
        )
        self.thread.start()

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def get_output(self):
        with self.condition:
            if self.output_queue:
                frame = self.output_queue.pop(0)
                self.condition.notify_all()
                return frame
        return None

    def _push(self, frame):
        with self.condition:
            # Back-pressure only. No frame dropping.
            while (
                self.running
                and len(self.output_queue) >= MAX_OUTPUT_QUEUE
            ):
                self.condition.wait(timeout=0.05)

            if self.running:
                self.output_queue.append(frame)
                self.condition.notify_all()

    def _render(self, frame):
        output = frame.copy()

        for track in list(
            self.tracker.active_tracks.values()
        ):
            reveal = (
                track.whitelisted
                or self.privacy.persistent_reveal(track)
            )

            if not reveal:
                apply_emoji(output, track.bbox)

        return output

    def _loop(self):
        processed_segments = set()
        frame_no = 0

        while self.running:
            ready = self.camera_buffer.get_ready_segments()

            segment = next(
                (
                    s for s in ready
                    if s.sequence not in processed_segments
                ),
                None
            )

            if segment is None:
                time.sleep(0.005)
                continue

            cap = cv2.VideoCapture(segment.path)

            if not cap.isOpened():
                self.camera_buffer.remove_segment(segment.sequence)
                processed_segments.add(segment.sequence)
                continue

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Submit every processed video frame to the AI worker.
                # The worker keeps the newest frame and runs as fast as
                # the GPU/model allows. Video frames themselves are not
                # discarded from the delayed stream.
                self.ai.submit(frame)

                result = self.ai.get_result()
                self.tracker.step(result, time.time())

                output = self._render(frame)
                self._push(output)

                frame_no += 1
                self.count += 1

                elapsed = time.perf_counter() - self.timer
                if elapsed >= 1.0:
                    self.processing_fps = self.count / elapsed
                    self.count = 0
                    self.timer = time.perf_counter()

            cap.release()
            processed_segments.add(segment.sequence)
            self.camera_buffer.remove_segment(segment.sequence)


# ---------------- MAIN ----------------

def main():
    print("=" * 70)
    print("PrivaStream - SAMPLE 10")
    print("YOLO face detection + ArcFace recognition")
    print("30s delayed privacy output / 60 FPS target")
    print("=" * 70)

    init_emoji()

    yolo, yolo_device = load_yolo()
    arcface, arcface_provider = load_arcface()

    ai = AsyncAIWorker(
        yolo,
        yolo_device,
        arcface
    )
    tracker = HybridTracker()
    privacy = PrivacyManager()

    camera_buffer = TemporalCameraBuffer(
        os.path.join(
            os.environ.get("TEMP", "."),
            "PrivaStreamBuffer"
        )
    )

    camera_buffer.start(0)

    while (
        camera_buffer.width == 0
        and camera_buffer.running
    ):
        time.sleep(0.05)

    if camera_buffer.error:
        camera_buffer.stop()
        raise RuntimeError(camera_buffer.error)

    live_window = "PrivaStream - LIVE CAMERA"
    output_window = "PrivaStream - PROCESSED OUTPUT"

    cv2.namedWindow(live_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(output_window, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(
        live_window,
        camera_buffer.width // 2,
        camera_buffer.height // 2
    )
    cv2.resizeWindow(
        output_window,
        camera_buffer.width,
        camera_buffer.height
    )

    cv2.setMouseCallback(
        live_window,
        privacy.live_callback
    )
    cv2.setMouseCallback(
        output_window,
        privacy.output_callback
    )

    try:
        vcam = pyvirtualcam.Camera(
            width=camera_buffer.width,
            height=camera_buffer.height,
            fps=TARGET_FPS,
            fmt=pyvirtualcam.PixelFormat.BGR
        )
        print("[VCAM] Virtual camera:", vcam.device)
    except Exception as e:
        camera_buffer.stop()
        cv2.destroyAllWindows()
        raise RuntimeError(f"Virtual camera failed: {e}")

    ai.start()

    processor = StreamProcessor(
        camera_buffer,
        ai,
        tracker,
        privacy
    )
    processor.start()

    print()
    print("[CONTROLS]")
    print("  LIVE click  = enroll PRIMARY")
    print("  OUTPUT click = toggle bystander reveal/mask")
    print("  R = reset identities")
    print("  Q = quit")
    print()

    next_output = time.perf_counter()
    interval = 1.0 / TARGET_FPS
    output_started = False

    try:
        while camera_buffer.running:
            # -------- LIVE --------
            live = camera_buffer.get_live_frame()

            if live is not None:
                live_display = live.copy()

                cv2.rectangle(
                    live_display,
                    (10, 10),
                    (500, 65),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    live_display,
                    "LIVE | CLICK FACE TO SELECT PRIMARY",
                    (20, 47),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )
                cv2.imshow(live_window, live_display)

                privacy.enroll_primary(
                    ai.get_result(),
                    privacy.consume_live_click(),
                    ai
                )

            # -------- OUTPUT CLICK --------
            track = clicked_track(
                tracker,
                privacy.consume_output_click()
            )
            if track is not None:
                privacy.toggle_track(track)

            # -------- OUTPUT --------
            output = processor.get_output()

            if output is not None:
                output_started = True

                display = output.copy()

                hud = [
                    "PRIVASTREAM | PRIVACY ACTIVE",
                    f"OUTPUT: {TARGET_FPS:.0f} FPS",
                    f"DELAY: {BUFFER_DELAY_SECONDS:.0f} SEC",
                    f"YOLO AI: {ai.fps:.1f} FPS",
                    f"YOLO/AI LATENCY: {ai.latency_ms:.1f} ms",
                    f"PROCESSOR: {processor.processing_fps:.1f} FPS",
                    f"ARCFACE: {arcface_provider}",
                    f"FACES: {len(tracker.active_tracks)}"
                ]

                cv2.rectangle(
                    display,
                    (10, 10),
                    (430, 195),
                    (0, 0, 0),
                    -1
                )

                y = 34
                for line in hud:
                    cv2.putText(
                        display,
                        line,
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (220, 220, 220),
                        1
                    )
                    y += 22

                cv2.imshow(output_window, display)
                vcam.send(output)

            elif not output_started:
                wait = np.zeros(
                    (
                        camera_buffer.height,
                        camera_buffer.width,
                        3
                    ),
                    dtype=np.uint8
                )

                buffered = camera_buffer.buffered_seconds()
                progress = min(
                    buffered / BUFFER_DELAY_SECONDS,
                    1.0
                )

                cv2.putText(
                    wait,
                    "PRIVASTREAM - PREPARING DELAYED OUTPUT",
                    (40, camera_buffer.height // 2 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (220, 220, 220),
                    2
                )

                cv2.putText(
                    wait,
                    f"Privacy buffer: {buffered:.0f}/"
                    f"{BUFFER_DELAY_SECONDS:.0f} sec",
                    (40, camera_buffer.height // 2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (220, 220, 220),
                    2
                )

                bar_w = 500
                filled = int(bar_w * progress)

                cv2.rectangle(
                    wait,
                    (40, camera_buffer.height // 2 + 50),
                    (
                        40 + bar_w,
                        camera_buffer.height // 2 + 70
                    ),
                    (80, 80, 80),
                    2
                )

                if filled:
                    cv2.rectangle(
                        wait,
                        (40, camera_buffer.height // 2 + 50),
                        (
                            40 + filled,
                            camera_buffer.height // 2 + 70
                        ),
                        (0, 255, 0),
                        -1
                    )

                cv2.imshow(output_window, wait)

                # Safe placeholder only; raw camera is never sent.
                vcam.send(wait)

            # -------- CONTROLS --------
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                privacy.reset(tracker, ai)

            # -------- 60 FPS CLOCK --------
            next_output += interval
            sleep_time = next_output - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_output = time.perf_counter()

    except KeyboardInterrupt:
        pass

    finally:
        print("[SYSTEM] Stopping...")
        processor.stop()
        ai.stop()
        camera_buffer.stop()

        try:
            vcam.close()
        except Exception:
            pass

        cv2.destroyAllWindows()
        print("[SYSTEM] PrivaStream stopped.")


if __name__ == "__main__":
    main()
