import cv2
import numpy as np
import pyvirtualcam
import threading
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import onnxruntime as ort
from insightface.app import FaceAnalysis


# ============================================================
# 1. CONFIGURATION & RESOLUTION PRESETS
# ============================================================

# Target processing resolution presets:
# - "720p": (1280, 720) - Standard HD
# - "540p": (960, 540)  - High-performance balanced
# - "360p": (640, 360)  - Maximum framerate mode
RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "540p": (960, 540),
    "360p": (640, 360),
}

ACTIVE_PRESET = "720p"
CAMERA_WIDTH, CAMERA_HEIGHT = RESOLUTION_PRESETS[ACTIVE_PRESET]
TARGET_FPS = 60

# Model settings
MODEL_NAME = "buffalo_sc"

# Identity similarity threshold (cosine similarity >= 0.44 = primary user)
SIMILARITY_THRESHOLD = 0.44

# Asynchronous AI Worker target inference frequency (runs in background)
AI_TARGET_FPS = 8.0

# Fail-safe timeout: If AI has not produced a fresh result in this time (seconds),
# activate fail-safe privacy mode (blur all faces conservatively).
FAILSAFE_TIMEOUT = 0.75

# Identity re-verification safety timeout:
# A primary user track must be re-confirmed by AI within this time window.
# If not re-confirmed, it reverts to unknown to prevent identity hijacking.
PRIMARY_REVERIFY_TIMEOUT = 1.2

# Tracking parameters
MAX_MATCH_DISTANCE = 160.0  # Max pixel distance for associating track with detection
MAX_LOST_FRAMES = 25        # How many frames a lost track persists before removal
VELOCITY_SMOOTHING = 0.35   # Exponential smoothing factor for velocity


# ============================================================
# 2. DATA STRUCTURES
# ============================================================

@dataclass
class FaceDetection:
    bbox: np.ndarray          # [x1, y1, x2, y2]
    embedding: np.ndarray     # 512-d normalized face feature vector
    whitelisted: bool         # Matches enrolled primary user
    similarity: float         # Cosine similarity score


@dataclass
class AIResult:
    seq_id: int               # Monotonically increasing sequence number
    timestamp: float          # Time when inference completed
    latency_ms: float         # Exact duration of AI inference in milliseconds
    detections: List[FaceDetection] = field(default_factory=list)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    velocity: np.ndarray      # [vx, vy] per frame
    whitelisted: bool         # Currently deemed primary user
    lost_frames: int          # Consecutive frames without matching AI detection
    last_verified_time: float # Timestamp of last positive AI identity verification
    confidence: float         # Match confidence / similarity


# ============================================================
# 3. INSIGHTFACE & EXECUTION PROVIDER SETUP
# ============================================================

def initialize_insightface():
    """Initializes InsightFace with CUDA if available, falling back to CPU."""
    available_providers = ort.get_available_providers()
    print(f"[INFO] Available ONNX Providers: {available_providers}")

    preferred_providers = []
    if "CUDAExecutionProvider" in available_providers:
        preferred_providers.append("CUDAExecutionProvider")
    preferred_providers.append("CPUExecutionProvider")

    print(f"[INFO] Initializing InsightFace with providers: {preferred_providers}")
    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=preferred_providers
    )
    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    # Detect active execution provider
    active_provider = "CPUExecutionProvider"
    try:
        for model in app.models.values():
            if hasattr(model, "session"):
                providers = model.session.get_providers()
                if providers:
                    active_provider = providers[0]
                    break
    except Exception:
        pass

    print(f"[INFO] InsightFace loaded. Active Provider: {active_provider}")
    return app, active_provider


# ============================================================
# 4. EMBEDDING MATCHING (PRIMARY USER WHITELIST)
# ============================================================

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two normalized feature vectors."""
    dot = np.dot(vec1, vec2)
    norm = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-6
    return float(dot / norm)


def check_identity(embedding: np.ndarray, whitelist: List[np.ndarray], threshold: float) -> Tuple[bool, float]:
    """Compares face embedding against all enrolled primary user embeddings."""
    if not whitelist:
        return False, 0.0

    best_similarity = 0.0
    for saved in whitelist:
        sim = compute_cosine_similarity(embedding, saved)
        if sim > best_similarity:
            best_similarity = sim

    is_match = best_similarity >= threshold
    return is_match, best_similarity


# ============================================================
# 5. ASYNCHRONOUS AI WORKER (DECOUPLED FROM VIDEO LOOP)
# ============================================================

class AsyncAIWorker:
    """
    Runs InsightFace detection and embedding recognition in a background thread.
    Consumes the newest available frame ('latest frame wins') and publishes
    versioned AI results with a monotonic sequence ID.
    """
    def __init__(self, app: FaceAnalysis, target_fps: float = 8.0):
        self.app = app
        self.target_interval = 1.0 / max(1.0, target_fps)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Thread-safe exchange buffers
        self._lock = threading.Lock()
        self._latest_input_frame: Optional[np.ndarray] = None
        self._whitelist_embeddings: List[np.ndarray] = []
        self._similarity_threshold = SIMILARITY_THRESHOLD

        # Published output result
        self._seq_counter = 0
        self._latest_result = AIResult(
            seq_id=0,
            timestamp=0.0,
            latency_ms=0.0,
            detections=[]
        )

        # Worker metrics
        self.fps = 0.0
        self.latency_ms = 0.0
        self._frames_counted = 0
        self._fps_timer = time.perf_counter()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="AI-Worker")
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def submit_frame(self, frame: np.ndarray):
        """Called by the main loop to push the latest camera frame without blocking."""
        with self._lock:
            # Overwrite with newest frame (drop stale queued frames)
            self._latest_input_frame = frame

    def update_whitelist(self, whitelist: List[np.ndarray], threshold: float):
        """Updates enrolled primary user embeddings."""
        with self._lock:
            self._whitelist_embeddings = [w.copy() for w in whitelist]
            self._similarity_threshold = threshold

    def get_latest_result(self) -> AIResult:
        """Called by the main loop to get the newest published AI result."""
        with self._lock:
            return self._latest_result

    def _worker_loop(self):
        while self.running:
            loop_start = time.perf_counter()

            # Grab latest frame copy under lock
            frame_to_process = None
            current_whitelist = []
            thresh = SIMILARITY_THRESHOLD

            with self._lock:
                if self._latest_input_frame is not None:
                    frame_to_process = self._latest_input_frame.copy()
                    self._latest_input_frame = None  # Consume frame
                current_whitelist = self._whitelist_embeddings
                thresh = self._similarity_threshold

            if frame_to_process is None:
                time.sleep(0.005)
                continue

            # Run InsightFace Inference
            infer_start = time.perf_counter()
            detections: List[FaceDetection] = []

            try:
                faces = self.app.get(frame_to_process)
                for face in faces:
                    bbox = np.array(face.bbox, dtype=np.float32)
                    emb = face.embedding
                    is_primary, similarity = check_identity(emb, current_whitelist, thresh)
                    detections.append(FaceDetection(
                        bbox=bbox,
                        embedding=emb,
                        whitelisted=is_primary,
                        similarity=similarity
                    ))
            except Exception as e:
                print(f"[AI WORKER ERROR] {e}")

            infer_end = time.perf_counter()
            latency_ms = (infer_end - infer_start) * 1000.0

            # Publish versioned AIResult
            with self._lock:
                self._seq_counter += 1
                self._latest_result = AIResult(
                    seq_id=self._seq_counter,
                    timestamp=time.time(),
                    latency_ms=latency_ms,
                    detections=detections
                )
                self.latency_ms = latency_ms
                self._frames_counted += 1

            # Calculate AI worker FPS
            now = time.perf_counter()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self.fps = self._frames_counted / elapsed
                self._frames_counted = 0
                self._fps_timer = now

            # Pacing to avoid unnecessary CPU/GPU saturation
            elapsed_loop = time.perf_counter() - loop_start
            sleep_duration = self.target_interval - elapsed_loop
            if sleep_duration > 0:
                time.sleep(sleep_duration)


# ============================================================
# 6. DECOUPLED REAL-TIME TRACKER WITH IDENTITY SAFETY
# ============================================================

class SafeRealTimeTracker:
    """
    Lightweight, high-framerate face tracker designed for 60 FPS video loops.
    
    Architecture:
    - Ingests AI detections ONLY when seq_id is newly incremented (avoids jitter).
    - Smoothly predicts positions and decays velocity between AI detections.
    - Implements strict identity safety:
      * Demotes PRIMARY if identity is not re-verified within PRIMARY_REVERIFY_TIMEOUT.
      * Prevents new/unverified tracks from inheriting PRIMARY identity.
    """
    def __init__(self, max_match_dist: float = MAX_MATCH_DISTANCE, max_lost: int = MAX_LOST_FRAMES):
        self.max_match_dist = max_match_dist
        self.max_lost = max_lost
        self.active_tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.last_processed_seq = -1

    def step(self, ai_result: AIResult, current_time: float, dt: float):
        """
        Main update method called on every video frame (60 FPS).
        Decides whether to run full detection calibration or pure motion prediction.
        """
        if ai_result.seq_id > self.last_processed_seq:
            # New AI result arrived: perform association & state calibration
            self._calibrate_with_ai(ai_result, current_time)
            self.last_processed_seq = ai_result.seq_id
        else:
            # No new AI result: predict motion forward smoothly
            self._predict_motion(dt, current_time)

    def _calibrate_with_ai(self, ai_result: AIResult, current_time: float):
        matched_track_ids = set()

        for det in ai_result.detections:
            det_bbox = det.bbox
            det_center = (det_bbox[:2] + det_bbox[2:]) * 0.5

            best_track_id = None
            best_dist = self.max_match_dist

            # Find closest existing track
            for track_id, track in self.active_tracks.items():
                if track_id in matched_track_ids:
                    continue

                # Projected center using current velocity
                pred_bbox = track.bbox.copy()
                pred_bbox[[0, 2]] += track.velocity[0]
                pred_bbox[[1, 3]] += track.velocity[1]
                pred_center = (pred_bbox[:2] + pred_bbox[2:]) * 0.5

                dist = np.linalg.norm(det_center - pred_center)
                if dist < best_dist:
                    best_dist = dist
                    best_track_id = track_id

            if best_track_id is not None:
                # Update matched track
                track = self.active_tracks[best_track_id]
                old_center = (track.bbox[:2] + track.bbox[2:]) * 0.5
                movement = det_center - old_center

                # Smooth velocity estimation
                track.velocity = (
                    (1.0 - VELOCITY_SMOOTHING) * track.velocity +
                    VELOCITY_SMOOTHING * movement
                )
                track.bbox = det_bbox.copy()
                track.lost_frames = 0
                track.confidence = det.similarity

                # Identity safety update:
                # Only maintain/set whitelisted if this AI detection specifically verified it!
                if det.whitelisted:
                    track.whitelisted = True
                    track.last_verified_time = current_time
                else:
                    track.whitelisted = False

                matched_track_ids.add(best_track_id)
            else:
                # Spawn new track
                self.active_tracks[self.next_track_id] = Track(
                    track_id=self.next_track_id,
                    bbox=det_bbox.copy(),
                    velocity=np.zeros(2, dtype=np.float32),
                    whitelisted=det.whitelisted,
                    lost_frames=0,
                    last_verified_time=current_time if det.whitelisted else 0.0,
                    confidence=det.similarity
                )
                matched_track_ids.add(self.next_track_id)
                self.next_track_id += 1

        # Handle tracks that were not detected in this AI frame
        dead_tracks = []
        for track_id, track in self.active_tracks.items():
            if track_id not in matched_track_ids:
                track.lost_frames += 1
                # Project forward with velocity
                track.bbox[[0, 2]] += track.velocity[0]
                track.bbox[[1, 3]] += track.velocity[1]
                track.velocity *= 0.90  # Dampen velocity on missed detection

                # Strict identity safety: expire whitelisted status if unverified
                if track.whitelisted and (current_time - track.last_verified_time > PRIMARY_REVERIFY_TIMEOUT):
                    track.whitelisted = False

                if track.lost_frames > self.max_lost:
                    dead_tracks.append(track_id)

        for tid in dead_tracks:
            del self.active_tracks[tid]

    def _predict_motion(self, dt: float, current_time: float):
        """Smoothly moves all active tracks during intermediate frames between AI updates."""
        dead_tracks = []
        for track_id, track in self.active_tracks.items():
            # Apply velocity step
            track.bbox[[0, 2]] += track.velocity[0]
            track.bbox[[1, 3]] += track.velocity[1]
            track.velocity *= 0.98  # Gentle decay

            # Identity safety timeout check
            if track.whitelisted and (current_time - track.last_verified_time > PRIMARY_REVERIFY_TIMEOUT):
                track.whitelisted = False

        for tid in dead_tracks:
            del self.active_tracks[tid]

    def reset(self):
        self.active_tracks.clear()
        self.last_processed_seq = -1


# ============================================================
# 7. HIGH-PERFORMANCE GAUSSIAN BLUR & FEATHERING
# ============================================================

def apply_fast_feathered_blur(frame: np.ndarray, bbox: np.ndarray):
    """
    Ultra-fast, single-pass Gaussian privacy blur with feathered elliptical edges.
    
    Optimization techniques:
    1. Downsample ROI by 4x -> Gaussian blur -> Upsample (10x faster than raw large kernel).
    2. Vectorized 3-channel alpha blending in NumPy (no Python channel loops).
    3. In-place modification on frame memory to eliminate buffer allocations.
    """
    h, w, _ = frame.shape
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))

    roi_w = x2 - x1
    roi_h = y2 - y1

    if roi_w < 8 or roi_h < 8:
        return

    roi = frame[y1:y2, x1:x2]

    # Fast multi-scale anonymization blur:
    # Downsample by 4x, blur with small kernel, then scale back up
    ds_w = max(4, roi_w // 4)
    ds_h = max(4, roi_h // 4)
    small_roi = cv2.resize(roi, (ds_w, ds_h), interpolation=cv2.INTER_LINEAR)
    blurred_small = cv2.GaussianBlur(small_roi, (15, 15), 0)
    blurred_roi = cv2.resize(blurred_small, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    # Feathered elliptical alpha mask
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    center = (roi_w // 2, roi_h // 2)
    axes = (max(1, int(roi_w * 0.48)), max(1, int(roi_h * 0.48)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Smooth mask border
    feather_ksize = max(5, (min(roi_w, roi_h) // 6) | 1)
    mask = cv2.GaussianBlur(mask, (feather_ksize, feather_ksize), 0)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    # Vectorized in-place blending
    frame[y1:y2, x1:x2] = (
        roi.astype(np.float32) * (1.0 - alpha) +
        blurred_roi.astype(np.float32) * alpha
    ).astype(np.uint8)


# ============================================================
# 8. CLICK-TO-ENROLL PRIMARY USER
# ============================================================

class EnrollmentManager:
    def __init__(self):
        self.whitelist_embeddings: List[np.ndarray] = []
        self.clicked_coords: Optional[Tuple[int, int]] = None
        self.lock = threading.Lock()

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.lock:
                self.clicked_coords = (x, y)

    def process_click(self, tracker: SafeRealTimeTracker, latest_ai_result: AIResult, ai_worker: AsyncAIWorker):
        coords = None
        with self.lock:
            coords = self.clicked_coords
            self.clicked_coords = None

        if coords is None:
            return

        cx, cy = coords

        # Find clicked track
        for track_id, track in tracker.active_tracks.items():
            x1, y1, x2, y2 = track.bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                # Find matching embedding from latest AI detections
                for det in latest_ai_result.detections:
                    dx1, dy1, dx2, dy2 = det.bbox
                    if dx1 <= cx <= dx2 and dy1 <= cy <= dy2:
                        self.whitelist_embeddings.append(det.embedding.copy())
                        ai_worker.update_whitelist(self.whitelist_embeddings, SIMILARITY_THRESHOLD)
                        track.whitelisted = True
                        track.last_verified_time = time.time()
                        print(f"[ENROLL] Face enrolled as Primary User! (Track ID: {track_id})")
                        return

                print("[WARN] No fresh embedding available for clicked face. Try clicking again.")
                return

    def reset(self, tracker: SafeRealTimeTracker, ai_worker: AsyncAIWorker):
        self.whitelist_embeddings.clear()
        ai_worker.update_whitelist(self.whitelist_embeddings, SIMILARITY_THRESHOLD)
        tracker.reset()
        print("[INFO] All primary identities reset.")


# ============================================================
# 9. REAL-TIME METRICS
# ============================================================

class PerformanceMetrics:
    """Measures true real-time elapsed intervals without hardcoded or artificial values."""
    def __init__(self):
        self.camera_fps = 0.0
        self.output_fps = 0.0
        self._cam_frames = 0
        self._cam_timer = time.perf_counter()
        self._out_frames = 0
        self._out_timer = time.perf_counter()

    def tick_camera(self):
        self._cam_frames += 1
        now = time.perf_counter()
        elapsed = now - self._cam_timer
        if elapsed >= 0.75:
            self.camera_fps = self._cam_frames / elapsed
            self._cam_frames = 0
            self._cam_timer = now

    def tick_output(self):
        self._out_frames += 1
        now = time.perf_counter()
        elapsed = now - self._out_timer
        if elapsed >= 0.75:
            self.output_fps = self._out_frames / elapsed
            self._out_frames = 0
            self._out_timer = now


# ============================================================
# 10. MAIN PIPELINE EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print(" PrivaStream: Real-Time Asynchronous Privacy Pipeline")
    print("=" * 60)

    # 1. Initialize InsightFace & detect provider
    app, active_provider = initialize_insightface()

    # 2. Initialize Camera
    print(f"[INFO] Opening Camera at {CAMERA_WIDTH}x{CAMERA_HEIGHT} (Target: {TARGET_FPS} FPS)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow on Windows for fastest response
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_cam_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Camera Hardware Config: {actual_width}x{actual_height} @ {actual_cam_fps:.1f} reported FPS")

    # 3. Start Asynchronous AI Worker
    ai_worker = AsyncAIWorker(app=app, target_fps=AI_TARGET_FPS)
    ai_worker.start()
    print("[INFO] Asynchronous AI worker thread started.")

    # 4. Initialize Tracker & Enrollment Manager
    tracker = SafeRealTimeTracker()
    enrollment = EnrollmentManager()
    metrics = PerformanceMetrics()

    # 5. UI Setup
    window_name = "PrivaStream Local Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_width, actual_height)
    cv2.setMouseCallback(window_name, enrollment.on_mouse_click)

    # 6. Initialize Virtual Camera
    print("[INFO] Starting pyvirtualcam output stream...")
    try:
        vcam = pyvirtualcam.Camera(
            width=actual_width,
            height=actual_height,
            fps=TARGET_FPS,
            fmt=pyvirtualcam.PixelFormat.BGR
        )
        print(f"[INFO] Virtual Camera started on device: {vcam.device}")
    except Exception as e:
        print(f"[ERROR] Could not start virtual camera: {e}")
        ai_worker.stop()
        cap.release()
        return

    print("\n[CONTROLS]")
    print(" - Click on your face to enroll as PRIMARY user")
    print(" - Press 'r' to reset enrolled identities")
    print(" - Press 'q' to quit\n")

    prev_frame_time = time.perf_counter()

    try:
        while True:
            # Step A: Capture latest camera frame
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera frame capture failed.")
                break

            metrics.tick_camera()
            now = time.perf_counter()
            dt = max(0.001, now - prev_frame_time)
            prev_frame_time = now

            # Step B: Push frame to asynchronous AI worker (non-blocking)
            ai_worker.submit_frame(frame)

            # Step C: Retrieve latest published AI result
            latest_ai_result = ai_worker.get_latest_result()
            ai_age = time.time() - latest_ai_result.timestamp if latest_ai_result.timestamp > 0 else 999.0

            # Step D: Update Tracker (Calibrates on new seq_id, predicts smoothly on intermediate frames)
            tracker.step(latest_ai_result, current_time=time.time(), dt=dt)

            # Step E: Process click enrollment if requested
            enrollment.process_click(tracker, latest_ai_result, ai_worker)

            # Step F: Determine Fail-Safe Privacy Mode
            # If AI is stale or hasn't produced results yet, fail-safe triggers
            failsafe_active = (latest_ai_result.seq_id == 0) or (ai_age > FAILSAFE_TIMEOUT)

            # Step G: Single-Pass Privacy Filtering (In-place blur on output_frame)
            output_frame = frame.copy()

            for track_id, track in list(tracker.active_tracks.items()):
                # Strict privacy rule:
                # If fail-safe is active OR track is not whitelisted, blur face!
                if failsafe_active or (not track.whitelisted):
                    apply_fast_feathered_blur(output_frame, track.bbox)

            # Step H: Send processed frame to pyvirtualcam
            vcam.send(output_frame)

            # Step I: Render Local Preview HUD & Bounding Boxes
            # Clone output_frame only for local UI display annotations
            display_frame = output_frame.copy()

            # Draw track bounding boxes and status labels
            for track_id, track in tracker.active_tracks.items():
                x1, y1, x2, y2 = [int(v) for v in track.bbox]
                
                if failsafe_active:
                    color = (0, 140, 255)  # Orange: Fail-Safe
                    status_text = "FAIL-SAFE"
                elif track.whitelisted:
                    color = (0, 255, 0)    # Green: Primary User
                    status_text = f"PRIMARY ({track.confidence:.2f})"
                else:
                    color = (0, 0, 255)    # Red: Bystander Blurred
                    status_text = "BLURRED"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display_frame,
                    f"ID:{track_id} [{status_text}]",
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

            # Draw HUD Diagnostics
            metrics.tick_output()

            hud_y = 28
            hud_spacing = 24
            
            # Semi-transparent HUD background panel
            cv2.rectangle(display_frame, (10, 10), (340, 175), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (10, 10), (340, 175), (80, 80, 80), 1)

            # HUD readouts
            cv2.putText(display_frame, f"PIPELINE OUTPUT: {metrics.output_fps:.1f} FPS", (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            hud_y += hud_spacing

            cv2.putText(display_frame, f"CAMERA INPUT:   {metrics.camera_fps:.1f} FPS", (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            hud_y += hud_spacing

            cv2.putText(display_frame, f"AI INFERENCE:   {ai_worker.fps:.1f} FPS ({ai_worker.latency_ms:.1f}ms)", (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            hud_y += hud_spacing

            provider_label = "CUDA" if "CUDA" in active_provider else "CPU"
            cv2.putText(display_frame, f"ENGINE / MODE:  {provider_label} / {ACTIVE_PRESET}", (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            hud_y += hud_spacing

            privacy_status = "FAIL-SAFE ACTIVE" if failsafe_active else "PRIVACY ACTIVE"
            privacy_color = (0, 140, 255) if failsafe_active else (0, 255, 0)
            cv2.putText(display_frame, f"STATE: {privacy_status}", (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, privacy_color, 2)
            hud_y += hud_spacing

            faces_count = len(tracker.active_tracks)
            cv2.putText(display_frame, f"FACES TRACKED:  {faces_count}", (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Step J: Display Preview & Handle Keyboard Input
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quitting PrivaStream...")
                break
            elif key == ord('r'):
                enrollment.reset(tracker, ai_worker)

    finally:
        print("[INFO] Cleaning up resources...")
        ai_worker.stop()
        vcam.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] PrivaStream stopped cleanly.")


if __name__ == "__main__":

    main()