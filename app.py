import os
import sys

# Suppress ONNX runtime verbose logs and block problematic CUDA DLL probing
os.environ["ORT_DISABLE_PRELOAD_DLLS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["ORT_LOG_LEVEL"] = "3"

import cv2
import urllib.request
import numpy as np
import pyvirtualcam
import threading
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import onnxruntime as ort
from ultralytics import YOLO
from insightface.app import FaceAnalysis


# ============================================================
# 1. CONFIGURATION
# ============================================================

RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "540p": (960, 540),
    "360p": (640, 360),
}

ACTIVE_PRESET = "720p"
CAMERA_WIDTH, CAMERA_HEIGHT = RESOLUTION_PRESETS[ACTIVE_PRESET]

# Target rendering & virtual camera output FPS (Smooth 30 FPS matched to camera hardware)
TARGET_RENDER_FPS = 30.0

# Configurable AI detector inference rate (8.0 - 15.0 FPS, default ~10.0 FPS)
# Low inference rate keeps CPU lightweight while the tracker smoothly interpolates.
AI_TARGET_FPS = 10.0

# Downscaled width for fast YOLO inference
INFERENCE_WIDTH = 640

# Bounding box temporal smoothing factor (0.0 = no update, 1.0 = instant snap)
# 0.40 provides butter-smooth, jitter-free box interpolation toward new predictions.
BBOX_SMOOTHING = 0.40
VELOCITY_SMOOTHING = 0.35

YOLO_MODEL_CANDIDATES = [
    "yolov8n-face.pt",
    os.path.join("models", "yolov11n-face.pt"),
    os.path.join("models", "yolov8n-face.pt"),
]
YOLO_MODEL_PATH = "yolov8n-face.pt"
YOLO_CONF_THRESHOLD = 0.40
INSIGHTFACE_MODEL = "buffalo_s"
DATA_STORAGE_PATH = "enrollments.npz"

SIMILARITY_THRESHOLD = 0.44
REID_THRESHOLD = 0.48
MAX_MATCH_DISTANCE = 180.0
MAX_LOST_FRAMES = 45
MAX_PREDICT_FRAMES = 12

PRIMARY_REVERIFY_TIMEOUT = 1.5
FAILSAFE_TIMEOUT = 0.75
CAMERA_BUFFER_SIZE = 1


# ============================================================
# 2. WEIGHTS RESOLUTION
# ============================================================

def resolve_or_download_yolo_model() -> str:
    for candidate in YOLO_MODEL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate

    print(f"[INIT] Model '{YOLO_MODEL_PATH}' not found. Downloading weights mirror...")
    mirror_url = "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"
    try:
        urllib.request.urlretrieve(mirror_url, YOLO_MODEL_PATH)
        print(f"[INIT] Model downloaded successfully to {YOLO_MODEL_PATH}.")
        return YOLO_MODEL_PATH
    except Exception as e:
        print(f"[ERROR] Auto-download failed: {e}")
        return YOLO_MODEL_PATH


# ============================================================
# 3. DATA STRUCTURES
# ============================================================

@dataclass
class FaceDetection:
    bbox: np.ndarray
    embedding: np.ndarray
    whitelisted: bool
    similarity: float
    reused_identity: bool = False


@dataclass
class AIResult:
    seq_id: int = 0
    timestamp: float = 0.0
    latency_ms: float = 0.0
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
    last_update_time: float = 0.0
    manual_revealed: bool = False


# ============================================================
# 4. MATH & COSINE SIMILARITY
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def check_identity(embedding: np.ndarray, whitelist: List[np.ndarray], threshold: float) -> Tuple[bool, float]:
    if not whitelist or embedding is None or len(embedding) == 0:
        return False, 0.0

    best_sim = -1.0
    for saved in whitelist:
        sim = cosine_similarity(embedding, saved)
        if sim > best_sim:
            best_sim = sim
    return (best_sim >= threshold), best_sim


# ============================================================
# 5. INITIALIZE MODELS
# ============================================================

def initialize_models():
    model_path = resolve_or_download_yolo_model()
    print(f"[INIT] Loading YOLO-Face model: {model_path}...")
    yolo_model = YOLO(model_path)

    print(f"[INIT] Initializing InsightFace on CPU...")
    app = FaceAnalysis(
        name=INSIGHTFACE_MODEL,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    return yolo_model, app


# ============================================================
# 6. ASYNCHRONOUS AI WORKER (LATEST-FRAME ONLY, ON-DEMAND RECOGNITION)
# ============================================================

class AsyncAIWorker:
    """
    Runs YOLO face detection + On-Demand InsightFace ArcFace recognition in a background thread.
    - Always grabs the single MOST RECENT frame from the latest-frame slot.
    - Never processes a backlog; drops unexamined older frames.
    - Performs expensive ArcFace recognition ONLY when a new face appears or re-verification is due.
    - Publishes latest AI prediction for smooth consumption by the renderer.
    """
    def __init__(self, yolo_model: YOLO, face_app: FaceAnalysis, target_fps: float = AI_TARGET_FPS):
        self.yolo = yolo_model
        self.face_app = face_app
        self.target_interval = 1.0 / max(1.0, target_fps)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        self.latest_frame: Optional[np.ndarray] = None
        self.active_tracks_summary: List[Tuple[int, float, float, np.ndarray, float, bool, float]] = []
        self.whitelist_embeddings: List[np.ndarray] = []
        self.threshold = SIMILARITY_THRESHOLD

        self.latest_result = AIResult()
        self.sequence = 0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.recognition_count = 0
        self._count = 0
        self._fps_timer = time.perf_counter()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="PrivaStream-AI")
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def submit_frame(self, frame: np.ndarray, tracks_summary: Optional[List] = None):
        with self.lock:
            # Replaces any unexamined frame with the latest frame reference (ZERO backlog)
            self.latest_frame = frame
            if tracks_summary is not None:
                self.active_tracks_summary = tracks_summary

    def update_whitelist(self, embeddings: List[np.ndarray]):
        with self.lock:
            self.whitelist_embeddings = [e.copy() for e in embeddings]

    def get_latest_result(self) -> AIResult:
        with self.lock:
            return self.latest_result

    def _worker_loop(self):
        rec_model = self.face_app.models.get("recognition", None)

        while self.running:
            loop_start = time.perf_counter()
            frame = None
            tracks_info = []

            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame
                    self.latest_frame = None
                tracks_info = list(self.active_tracks_summary)
                whitelist = [e.copy() for e in self.whitelist_embeddings]
                threshold = self.threshold

            if frame is None:
                time.sleep(0.002)
                continue

            infer_start = time.perf_counter()
            detections = []
            now = time.time()

            try:
                orig_h, orig_w = frame.shape[:2]
                scale = INFERENCE_WIDTH / float(orig_w)
                infer_h = int(orig_h * scale)
                infer_frame = cv2.resize(frame, (INFERENCE_WIDTH, infer_h), interpolation=cv2.INTER_LINEAR)

                # 1. Fast YOLO Face Detection (~10-15ms)
                results = self.yolo.predict(infer_frame, conf=YOLO_CONF_THRESHOLD, verbose=False)

                if results and len(results) > 0:
                    yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
                    scale_inv = 1.0 / scale

                    # 2. Extract Embeddings ONLY when needed (skip for already tracked & verified faces)
                    for box in yolo_boxes:
                        x1 = int(box[0] * scale_inv)
                        y1 = int(box[1] * scale_inv)
                        x2 = int(box[2] * scale_inv)
                        y2 = int(box[3] * scale_inv)

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(orig_w, x2), min(orig_h, y2)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        det_cx = (x1 + x2) * 0.5
                        det_cy = (y1 + y2) * 0.5

                        # Check if this detection spatially matches a currently tracked face with valid embedding
                        matched_track = None
                        best_dist = float("inf")
                        for t_id, t_cx, t_cy, t_emb, t_last_verif, t_white, t_conf in tracks_info:
                            dist = float(np.hypot(det_cx - t_cx, det_cy - t_cy))
                            if dist < MAX_MATCH_DISTANCE and dist < best_dist:
                                best_dist = dist
                                matched_track = (t_id, t_emb, t_last_verif, t_white, t_conf)

                        # Decision: Can we reuse existing embedding, or must we run expensive ArcFace?
                        reused = False
                        if matched_track is not None:
                            _, t_emb, t_last_verif, t_white, t_conf = matched_track
                            # If face was verified recently and has a valid embedding, reuse it!
                            if (now - t_last_verif < PRIMARY_REVERIFY_TIMEOUT) and (np.linalg.norm(t_emb) > 1e-4):
                                embedding = t_emb
                                is_primary = t_white
                                similarity = t_conf
                                reused = True

                        if not reused:
                            # Run expensive ArcFace recognition ONLY for new/unverified faces
                            crop = frame[y1:y2, x1:x2]
                            embedding = None

                            if rec_model is not None and crop.size > 0:
                                try:
                                    face_img = cv2.resize(crop, (112, 112))
                                    feat = rec_model.get_feat(face_img)
                                    embedding = np.asarray(feat, dtype=np.float32).flatten()
                                    self.recognition_count += 1
                                except Exception:
                                    embedding = None

                            if embedding is None:
                                embedding = np.zeros(512, dtype=np.float32)
                            else:
                                norm = np.linalg.norm(embedding)
                                if norm > 1e-8:
                                    embedding /= norm

                            is_primary, similarity = check_identity(embedding, whitelist, threshold)

                        detections.append(
                            FaceDetection(
                                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                                embedding=embedding,
                                whitelisted=is_primary,
                                similarity=similarity,
                                reused_identity=reused
                            )
                        )

            except Exception as e:
                print(f"[AI WORKER ERROR] {e}")

            latency_ms = (time.perf_counter() - infer_start) * 1000.0

            with self.lock:
                self.sequence += 1
                self.latest_result = AIResult(
                    seq_id=self.sequence,
                    timestamp=time.time(),
                    latency_ms=latency_ms,
                    detections=detections
                )
                self.latency_ms = latency_ms

            self._count += 1
            now_perf = time.perf_counter()
            elapsed = now_perf - self._fps_timer
            if elapsed >= 1.0:
                self.fps = self._count / elapsed
                self._count = 0
                self._fps_timer = now_perf

            work_time = time.perf_counter() - loop_start
            remaining = self.target_interval - work_time
            if remaining > 0:
                time.sleep(remaining)


# ============================================================
# 7. HYBRID TRACKER & SMOOTH BOUNDING BOX INTERPOLATION
# ============================================================

class HybridTracker:
    """
    Associates AI detections with tracks and applies TEMPORAL SMOOTHING (BBOX_SMOOTHING).
    Gradually moves displayed bounding boxes toward new predictions to eliminate jitter,
    while smoothly extrapolating between detector runs.
    """
    def __init__(self):
        self.active_tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.last_ai_seq = -1

    @staticmethod
    def center(bbox: np.ndarray) -> np.ndarray:
        return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)

    def get_tracks_summary(self) -> List[Tuple[int, float, float, np.ndarray, float, bool, float]]:
        """Returns lightweight track summary to help AI worker avoid redundant recognition calls."""
        summary = []
        for t in self.active_tracks.values():
            c = self.center(t.bbox)
            summary.append((t.track_id, float(c[0]), float(c[1]), t.embedding, t.last_verified_time, t.whitelisted, t.confidence))
        return summary

    def step(self, ai_result: AIResult, now: float):
        if ai_result.seq_id > self.last_ai_seq:
            self._apply_ai_result(ai_result, now)
            self.last_ai_seq = ai_result.seq_id
        else:
            self._predict_one_frame(now)

    def _apply_ai_result(self, result: AIResult, now: float):
        detections = result.detections
        matched_tracks = set()
        matched_detections = set()
        candidates = []

        for di, detection in enumerate(detections):
            det_center = self.center(detection.bbox)
            for track_id, track in self.active_tracks.items():
                if track_id in matched_tracks:
                    continue

                dist = float(np.linalg.norm(det_center - self.center(track.bbox)))
                emb_sim = cosine_similarity(detection.embedding, track.embedding)

                spatial_ok = dist <= MAX_MATCH_DISTANCE
                reid_ok = emb_sim >= REID_THRESHOLD

                if not spatial_ok and not reid_ok:
                    continue

                emb_cost = 1.0 - max(-1.0, min(1.0, emb_sim))
                norm_dist = min(dist / max(1.0, MAX_MATCH_DISTANCE), 2.0)
                score = (emb_cost * 2.0) + norm_dist - (0.2 if spatial_ok else 0.0)
                candidates.append((score, di, track_id))

        candidates.sort(key=lambda x: x[0])

        for score, di, track_id in candidates:
            if di in matched_detections or track_id in matched_tracks:
                continue

            detection = detections[di]
            track = self.active_tracks[track_id]

            # Smooth temporal interpolation toward newest detected coordinates (eliminates jumping)
            target_bbox = detection.bbox.copy()
            track.bbox = (1.0 - BBOX_SMOOTHING) * track.bbox + BBOX_SMOOTHING * target_bbox

            movement = self.center(detection.bbox) - self.center(track.bbox)
            measured_vel = movement / max(1.0, track.lost_frames + 1.0)
            track.velocity = ((1.0 - VELOCITY_SMOOTHING) * track.velocity) + (VELOCITY_SMOOTHING * measured_vel)

            if not detection.reused_identity and np.linalg.norm(detection.embedding) > 1e-4:
                track.embedding = (0.70 * track.embedding) + (0.30 * detection.embedding)
                norm = np.linalg.norm(track.embedding)
                if norm > 1e-8:
                    track.embedding /= norm

            track.lost_frames = 0
            track.predict_frames = 0
            track.confidence = detection.similarity
            track.last_update_time = now

            if detection.whitelisted:
                track.whitelisted = True
                track.last_verified_time = now
            else:
                track.whitelisted = False

            matched_tracks.add(track_id)
            matched_detections.add(di)

        # New detections
        for di, detection in enumerate(detections):
            if di in matched_detections:
                continue

            new_id = self.next_track_id
            self.next_track_id += 1
            self.active_tracks[new_id] = Track(
                track_id=new_id,
                bbox=detection.bbox.copy(),
                embedding=detection.embedding.copy(),
                velocity=np.zeros(2, dtype=np.float32),
                whitelisted=detection.whitelisted,
                confidence=detection.similarity,
                last_verified_time=now if detection.whitelisted else 0.0,
                last_update_time=now
            )
            matched_tracks.add(new_id)

        self._age_and_prune(matched_tracks, now)

    def _predict_one_frame(self, now: float):
        self._age_and_prune(set(), now)

    def _age_and_prune(self, matched_tracks: set, now: float):
        dead = []
        for track_id, track in self.active_tracks.items():
            if track_id in matched_tracks:
                continue

            track.lost_frames += 1
            track.predict_frames += 1

            if track.whitelisted and (now - track.last_verified_time > PRIMARY_REVERIFY_TIMEOUT):
                track.whitelisted = False

            # Predict motion between AI inference cycles
            if track.predict_frames <= MAX_PREDICT_FRAMES:
                track.bbox[0] += track.velocity[0]
                track.bbox[2] += track.velocity[0]
                track.bbox[1] += track.velocity[1]
                track.bbox[3] += track.velocity[1]
                track.velocity *= 0.95
            else:
                track.velocity *= 0.50

            if track.lost_frames > MAX_LOST_FRAMES:
                dead.append(track_id)

        for track_id in dead:
            del self.active_tracks[track_id]

    def reset(self):
        self.active_tracks.clear()
        self.next_track_id = 0
        self.last_ai_seq = -1


# ============================================================
# 8. VECTORIZED EMOJI OVERLAY (CLEAN PRIVACY MASK)
# ============================================================

def _build_emoji_template() -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
        font_path = r"C:\Windows\Fonts\seguiemj.ttf"
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 256)
            img = Image.new("RGBA", (320, 320), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), "😎", font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            draw.text(((320 - w) // 2 - bbox[0], (320 - h) // 2 - bbox[1]), "😎", font=font, embedded_color=True)
            rgba = np.asarray(img, dtype=np.uint8)
            # Convert RGBA to BGRA for correct color rendering in OpenCV
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    except Exception:
        pass

    # Built-in geometric vector avatar fallback (BGRA format)
    template = np.zeros((320, 320, 4), dtype=np.uint8)
    cv2.circle(template, (160, 160), 140, (0, 215, 255, 255), -1)
    cv2.circle(template, (110, 125), 25, (0, 0, 0, 255), -1)
    cv2.circle(template, (210, 125), 25, (0, 0, 0, 255), -1)
    cv2.ellipse(template, (160, 200), (60, 35), 0, 0, 180, (0, 0, 0, 255), 10)
    return template

EMOJI_TEMPLATE = _build_emoji_template()


def apply_emoji_mask(frame: np.ndarray, bbox: np.ndarray):
    """High-speed vectorized in-place alpha blending."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))

    if x2 <= x1 or y2 <= y1:
        return

    pad_x = max(2, int((x2 - x1) * 0.12))
    pad_y = max(2, int((y2 - y1) * 0.12))

    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

    tw, th = x2 - x1, y2 - y1
    scale = min(tw / EMOJI_TEMPLATE.shape[1], th / EMOJI_TEMPLATE.shape[0])
    ew = max(1, int(EMOJI_TEMPLATE.shape[1] * scale))
    eh = max(1, int(EMOJI_TEMPLATE.shape[0] * scale))

    emoji = cv2.resize(EMOJI_TEMPLATE, (ew, eh), interpolation=cv2.INTER_NEAREST)
    px = x1 + (tw - ew) // 2
    py = y1 + (th - eh) // 2
    px2, py2 = min(w, px + ew), min(h, py + eh)

    if px >= px2 or py >= py2:
        return

    sx2, sy2 = px2 - px, py2 - py
    alpha = emoji[:sy2, :sx2, 3:4].astype(np.float32) / 255.0
    rgb = emoji[:sy2, :sx2, :3]

    roi = frame[py:py2, px:px2]
    cv2.convertScaleAbs(roi * (1.0 - alpha) + rgb * alpha, roi)


# ============================================================
# 9. ENROLLMENT MANAGER & PERSISTENCE
# ============================================================

class EnrollmentManager:
    def __init__(self, filepath: str = DATA_STORAGE_PATH):
        self.filepath = filepath
        self.whitelist_embeddings: List[np.ndarray] = []
        self.manual_revealed_embeddings: List[np.ndarray] = []
        self.clicked_coords: Optional[Tuple[int, int]] = None
        self.lock = threading.Lock()
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.filepath):
            return
        try:
            with np.load(self.filepath) as data:
                if "whitelist" in data and data["whitelist"].size > 0:
                    self.whitelist_embeddings = [emb.copy() for emb in data["whitelist"]]
                if "revealed" in data and data["revealed"].size > 0:
                    self.manual_revealed_embeddings = [emb.copy() for emb in data["revealed"]]
            print(f"[STORAGE] Loaded {len(self.whitelist_embeddings)} Primary faces & {len(self.manual_revealed_embeddings)} Saved Bystanders from disk.")
        except Exception as e:
            print(f"[STORAGE ERROR] Could not load '{self.filepath}': {e}")

    def save_data(self):
        try:
            np.savez_compressed(
                self.filepath,
                whitelist=np.array(self.whitelist_embeddings, dtype=np.float32) if self.whitelist_embeddings else np.empty((0, 512), dtype=np.float32),
                revealed=np.array(self.manual_revealed_embeddings, dtype=np.float32) if self.manual_revealed_embeddings else np.empty((0, 512), dtype=np.float32)
            )
            print(f"[STORAGE] Enrolled identities safely saved to {self.filepath}")
        except Exception as e:
            print(f"[STORAGE ERROR] Could not save data: {e}")

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.lock:
                self.clicked_coords = (x, y)

    def process_click(self, tracker: HybridTracker):
        """Toggle reveal/mask for bystander face clicked on screen."""
        with self.lock:
            coords = self.clicked_coords
            self.clicked_coords = None

        if coords is None:
            return

        cx, cy = coords
        clicked_track = None
        best_dist = float("inf")

        for track in tracker.active_tracks.values():
            x1, y1, x2, y2 = track.bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                dist = float(np.linalg.norm(np.array([cx, cy]) - tracker.center(track.bbox)))
                if dist < best_dist:
                    best_dist = dist
                    clicked_track = track

        if clicked_track is None or clicked_track.whitelisted:
            return

        persistent_idx = None
        best_sim = -1.0
        for i, emb in enumerate(self.manual_revealed_embeddings):
            sim = cosine_similarity(clicked_track.embedding, emb)
            if sim > best_sim:
                best_sim = sim
                persistent_idx = i

        currently_revealed = clicked_track.manual_revealed or (persistent_idx is not None and best_sim >= REID_THRESHOLD)

        if currently_revealed:
            clicked_track.manual_revealed = False
            if persistent_idx is not None and best_sim >= REID_THRESHOLD:
                self.manual_revealed_embeddings.pop(persistent_idx)
            print(f"[CLICK] Bystander face: MASKED.")
        else:
            clicked_track.manual_revealed = True
            self.manual_revealed_embeddings.append(clicked_track.embedding.copy())
            print(f"[CLICK] Bystander face: MANUALLY REVEALED.")

        self.save_data()

    def enroll_primary(self, tracker: HybridTracker, ai_worker: AsyncAIWorker):
        if not tracker.active_tracks:
            print("[ENROLL] No face found to enroll. Please ensure your face is visible in the camera frame.")
            return

        largest = max(tracker.active_tracks.values(), key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))

        if np.linalg.norm(largest.embedding) < 1e-4:
            print("[ENROLL] Waiting for face recognition model to produce embedding. Try pressing 'E' again.")
            return

        self.whitelist_embeddings.append(largest.embedding.copy())
        ai_worker.update_whitelist(self.whitelist_embeddings)
        largest.whitelisted = True
        largest.last_verified_time = time.time()
        self.save_data()
        print(f"[ENROLL] Successfully enrolled largest face as PRIMARY.")

    def reset(self, tracker: HybridTracker, ai_worker: AsyncAIWorker):
        self.whitelist_embeddings.clear()
        self.manual_revealed_embeddings.clear()
        ai_worker.update_whitelist(self.whitelist_embeddings)
        tracker.reset()
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except Exception:
                pass
        print("[ENROLL] All identities and saved records cleared from disk.")


# ============================================================
# 10. MAIN LOW-LATENCY PIPELINE (LIVE STREAMING & ASYNC AI)
# ============================================================

def main():
    print("=" * 70)
    print("PrivaStream - Low-Latency Live Streaming Pipeline for OBS")
    print("Async AI + On-Demand Recognition + 30 FPS pyvirtualcam")
    print("=" * 70)

    yolo_model, face_app = initialize_models()

    # Direct high-speed camera capture with minimal driver buffer (buffer size = 1)
    print(f"[CAMERA] Opening camera @ {CAMERA_WIDTH}x{CAMERA_HEIGHT}...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera device 0.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_RENDER_FPS)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or CAMERA_WIDTH
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or CAMERA_HEIGHT
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_RENDER_FPS
    print(f"[CAMERA] Initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

    tracker = HybridTracker()
    enrollment = EnrollmentManager()
    ai_worker = AsyncAIWorker(yolo_model, face_app, AI_TARGET_FPS)
    ai_worker.update_whitelist(enrollment.whitelist_embeddings)

    window_name = "PrivaStream - Real-Time Privacy Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_width, actual_height)
    cv2.setMouseCallback(window_name, enrollment.on_mouse_click)

    # Virtual camera output (pyvirtualcam sends directly to OBS virtual webcam)
    vcam = None
    try:
        vcam = pyvirtualcam.Camera(
            width=actual_width, height=actual_height, fps=TARGET_RENDER_FPS, fmt=pyvirtualcam.PixelFormat.BGR
        )
        print(f"[VCAM] Live Virtual Camera streaming active: {vcam.device} @ {TARGET_RENDER_FPS:.0f} FPS")
    except Exception as e:
        print(f"[VCAM WARNING] Virtual camera unavailable ({e}). Live preview window active.")

    ai_worker.start()
    print("\n[CONTROLS]:")
    print("  Click face -> Reveal / Mask toggle on bystander")
    print("  E          -> Enroll primary identity (saves to disk)")
    print("  R          -> Reset identities")
    print("  Q          -> Quit\n")

    # Metrics & Diagnostic timers
    camera_frames_count = 0
    stream_frames_count = 0
    fps_timer = time.perf_counter()
    diag_timer = time.perf_counter()
    camera_fps = 0.0
    stream_fps = 0.0
    frame_interval = 1.0 / TARGET_RENDER_FPS

    try:
        while True:
            loop_start = time.perf_counter()

            # 1. Capture newest camera frame (hardware driver buffer = 1)
            ret, frame = cap.read()
            if not ret:
                print("[CAMERA ERROR] Frame capture failed.")
                time.sleep(0.01)
                continue

            camera_frames_count += 1

            # 2. Submit newest frame to AI worker with active track info (non-blocking, zero backlog)
            ai_worker.submit_frame(frame, tracker.get_tracks_summary())

            # 3. Get latest available AI prediction (never waits for AI)
            ai_result = ai_worker.get_latest_result()
            now = time.time()
            ai_age = (now - ai_result.timestamp) if ai_result.timestamp > 0 else float("inf")

            # 4. Update tracker with smooth temporal interpolation (BBOX_SMOOTHING)
            tracker.step(ai_result, now)
            enrollment.process_click(tracker)

            failsafe_active = (ai_result.seq_id == 0 or ai_age > FAILSAFE_TIMEOUT)

            # 5. Create single output frame for privacy masking (zero redundant intermediate copies)
            output_frame = frame.copy()

            # Apply privacy mask (clean emoji masks only, NO face boxes/labels)
            for track in list(tracker.active_tracks.values()):
                persistent_reveal = any(
                    cosine_similarity(track.embedding, saved) >= REID_THRESHOLD
                    for saved in enrollment.manual_revealed_embeddings
                )

                if failsafe_active or (not track.whitelisted and not track.manual_revealed and not persistent_reveal):
                    apply_emoji_mask(output_frame, track.bbox)

            # 6. Stream directly to Virtual Camera (OBS / browser / video calls)
            if vcam:
                vcam.send(output_frame)

            stream_frames_count += 1

            # 7. Local preview & HUD (drawn in-place on output_frame AFTER vcam.send, 0 extra copies)
            now_perf = time.perf_counter()
            elapsed_sec = now_perf - fps_timer
            if elapsed_sec >= 1.0:
                camera_fps = camera_frames_count / elapsed_sec
                stream_fps = stream_frames_count / elapsed_sec
                camera_frames_count = 0
                stream_frames_count = 0
                fps_timer = now_perf

            e2e_latency_ms = (time.perf_counter() - loop_start) * 1000.0

            hud = f"STREAM: {stream_fps:.1f} FPS | CAM: {camera_fps:.1f} FPS | AI: {ai_worker.fps:.1f} FPS ({ai_worker.latency_ms:.1f}ms) | E2E: {e2e_latency_ms:.1f}ms"
            cv2.rectangle(output_frame, (10, 10), (int(len(hud) * 10.5) + 10, 42), (0, 0, 0), -1)
            cv2.putText(output_frame, hud, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

            cv2.imshow(window_name, output_frame)

            # Periodic console diagnostics (printed once every 3 seconds)
            if now_perf - diag_timer >= 3.0:
                print(f"[DIAGNOSTICS] Stream/OBS: {stream_fps:.1f} FPS | Camera: {camera_fps:.1f} FPS | AI: {ai_worker.fps:.1f} FPS ({ai_worker.latency_ms:.1f}ms) | E2E Latency: {e2e_latency_ms:.1f}ms | Active Tracks: {len(tracker.active_tracks)}")
                diag_timer = now_perf

            # 8. Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                enrollment.reset(tracker, ai_worker)
            elif key == ord("e"):
                enrollment.enroll_primary(tracker, ai_worker)

            # 9. Frame pacing (sync with virtual camera clock or target interval)
            if vcam:
                vcam.sleep_until_next_frame()
            else:
                elapsed = time.perf_counter() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

    finally:
        print("\n[SYSTEM] Shutting down...")
        ai_worker.stop()
        if vcam:
            try:
                vcam.close()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Clean shutdown completed.")


if __name__ == "__main__":
    main()