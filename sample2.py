import cv2
import numpy as np
import pyvirtualcam
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import onnxruntime as ort
from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURATION
# ============================================================

RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "540p": (960, 540),
    "360p": (640, 360),
}

ACTIVE_PRESET = "720p"

CAMERA_WIDTH, CAMERA_HEIGHT = RESOLUTION_PRESETS[ACTIVE_PRESET]

TARGET_FPS = 60

# InsightFace model
MODEL_NAME = "buffalo_sc"

# Identity matching
SIMILARITY_THRESHOLD = 0.44

# AI inference rate.
# This does NOT control the video FPS.
AI_TARGET_FPS = 8.0

# If the AI result becomes older than this,
# the system enters privacy fail-safe mode.
FAILSAFE_TIMEOUT = 0.75

# Maximum time a primary identity can remain trusted
# without a fresh AI verification.
PRIMARY_REVERIFY_TIMEOUT = 1.2

# Tracker settings
MAX_MATCH_DISTANCE = 160.0
MAX_LOST_FRAMES = 30

# Velocity smoothing
VELOCITY_SMOOTHING = 0.35

# Blur settings
BLUR_DOWNSAMPLE = 4


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class FaceDetection:
    bbox: np.ndarray
    embedding: np.ndarray
    whitelisted: bool
    similarity: float


@dataclass
class AIResult:
    seq_id: int
    timestamp: float
    latency_ms: float
    detections: List[FaceDetection]
    success: bool = True


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    velocity: np.ndarray

    whitelisted: bool = False

    lost_frames: int = 0

    # Last time this face was positively verified by AI
    last_verified_time: float = 0.0

    # Tracker confidence
    confidence: float = 1.0


# ============================================================
# GLOBAL / MODEL STATE
# ============================================================

app = None


# ============================================================
# IDENTITY UTILITIES
# ============================================================

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two face embeddings.
    """
    denominator = (
        np.linalg.norm(vec1) *
        np.linalg.norm(vec2)
        + 1e-6
    )

    return float(np.dot(vec1, vec2) / denominator)


def find_best_whitelist_match(
    face_embedding: np.ndarray,
    whitelist_embeddings: List[np.ndarray]
) -> Tuple[bool, float]:
    """
    Compare a face embedding against all enrolled embeddings.
    Returns:
        (is_whitelisted, best_similarity)
    """

    if not whitelist_embeddings:
        return False, 0.0

    best_similarity = 0.0

    for saved_embedding in whitelist_embeddings:
        similarity = compute_cosine_similarity(
            face_embedding,
            saved_embedding
        )

        if similarity > best_similarity:
            best_similarity = similarity

    return (
        best_similarity >= SIMILARITY_THRESHOLD,
        best_similarity
    )


# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def get_centroid(bbox: np.ndarray) -> np.ndarray:
    """
    Return center point of bounding box.
    """

    return np.array([
        (bbox[0] + bbox[2]) / 2.0,
        (bbox[1] + bbox[3]) / 2.0
    ], dtype=np.float32)


def clip_bbox(
    bbox: np.ndarray,
    width: int,
    height: int
) -> np.ndarray:
    """
    Keep bounding box inside frame boundaries.
    """

    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))

    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float32
    )


# ============================================================
# FAST PRIVACY BLUR
# ============================================================

def apply_fast_feathered_blur(
    frame: np.ndarray,
    bbox: np.ndarray
):
    """
    Fast privacy blur.

    Instead of performing an expensive large Gaussian blur
    on the full-resolution ROI, the ROI is:

        1. Cropped
        2. Downsampled
        3. Blurred
        4. Upscaled
        5. Feather-blended using an elliptical mask

    This keeps the privacy effect visually smooth while
    significantly reducing the amount of expensive blur work.
    """

    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = [
        int(v) for v in bbox
    ]

    # Clip coordinates
    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(0, min(frame_width, x2))
    y2 = max(0, min(frame_height, y2))

    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]

    roi_h, roi_w = roi.shape[:2]

    if roi_w < 6 or roi_h < 6:
        return

    # --------------------------------------------------------
    # Downsample
    # --------------------------------------------------------

    small_w = max(2, roi_w // BLUR_DOWNSAMPLE)
    small_h = max(2, roi_h // BLUR_DOWNSAMPLE)

    small = cv2.resize(
        roi,
        (small_w, small_h),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # Blur
    # --------------------------------------------------------

    blurred_small = cv2.GaussianBlur(
        small,
        (15, 15),
        0
    )

    # --------------------------------------------------------
    # Upscale
    # --------------------------------------------------------

    blurred_roi = cv2.resize(
        blurred_small,
        (roi_w, roi_h),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # Elliptical feather mask
    # --------------------------------------------------------

    mask = np.zeros(
        (roi_h, roi_w),
        dtype=np.uint8
    )

    center = (
        roi_w // 2,
        roi_h // 2
    )

    axes = (
        max(1, roi_w // 2),
        max(1, roi_h // 2)
    )

    cv2.ellipse(
        mask,
        center,
        axes,
        0,
        0,
        360,
        255,
        -1
    )

    # Keep feathering kernel reasonably small
    min_dim = min(roi_w, roi_h)

    feather_size = min(
        21,
        max(5, (min_dim // 6) | 1)
    )

    mask = cv2.GaussianBlur(
        mask,
        (feather_size, feather_size),
        0
    )

    alpha = mask.astype(np.float32) / 255.0

    # --------------------------------------------------------
    # Vectorized blend
    # --------------------------------------------------------

    alpha = alpha[..., None]

    blended = (
        roi.astype(np.float32) * (1.0 - alpha)
        +
        blurred_roi.astype(np.float32) * alpha
    )

    roi[:] = blended.astype(np.uint8)


# ============================================================
# ASYNCHRONOUS AI WORKER
# ============================================================

class AsyncAIWorker:
    """
    Runs InsightFace in a background thread.

    IMPORTANT:

    The worker does NOT maintain a queue.

    It always processes the newest frame available.

    This prevents old frames from accumulating and creating
    latency.
    """

    def __init__(
        self,
        face_app: FaceAnalysis,
        target_fps: float
    ):
        self.app = face_app
        self.target_fps = target_fps

        self._lock = threading.Lock()

        self._latest_frame = None
        self._latest_frame_seq = 0

        self._latest_result: Optional[AIResult] = None

        self._whitelist_embeddings: List[np.ndarray] = []

        self._running = False
        self._thread = None

        # Metrics
        self.ai_fps = 0.0
        self.last_latency_ms = 0.0

        self._result_count = 0
        self._metrics_start = time.perf_counter()

    # --------------------------------------------------------
    # Start / stop
    # --------------------------------------------------------

    def start(self):
        self._running = True

        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )

        self._thread.start()

    def stop(self):
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # --------------------------------------------------------
    # Frame submission
    # --------------------------------------------------------

    def submit_frame(
        self,
        frame: np.ndarray
    ):
        """
        Submit the newest camera frame.

        No frame queue is created.
        Older frames are automatically discarded.
        """

        with self._lock:
            self._latest_frame = frame
            self._latest_frame_seq += 1

    # --------------------------------------------------------
    # Whitelist
    # --------------------------------------------------------

    def set_whitelist(
        self,
        embeddings: List[np.ndarray]
    ):
        with self._lock:
            # Replace list rather than modifying it in-place.
            self._whitelist_embeddings = list(embeddings)

    # --------------------------------------------------------
    # Latest result
    # --------------------------------------------------------

    def get_latest_result(self) -> Optional[AIResult]:
        with self._lock:
            return self._latest_result

    # --------------------------------------------------------
    # Worker
    # --------------------------------------------------------

    def _worker_loop(self):

        target_interval = 1.0 / self.target_fps

        next_inference_time = time.perf_counter()

        while self._running:

            now = time.perf_counter()

            # Maintain requested AI interval
            if now < next_inference_time:
                time.sleep(
                    min(
                        0.005,
                        next_inference_time - now
                    )
                )
                continue

            next_inference_time = now + target_interval

            # ------------------------------------------------
            # Get newest frame
            # ------------------------------------------------

            with self._lock:

                if self._latest_frame is None:
                    continue

                frame_to_process = self._latest_frame.copy()
                frame_seq = self._latest_frame_seq

                # Clear reference so we know whether a newer
                # frame arrived while inference was running.
                self._latest_frame = None

                whitelist = list(
                    self._whitelist_embeddings
                )

            # ------------------------------------------------
            # Run InsightFace
            # ------------------------------------------------

            inference_start = time.perf_counter()

            detections = []

            try:

                faces = self.app.get(
                    frame_to_process
                )

                for face in faces:

                    bbox = np.asarray(
                        face.bbox,
                        dtype=np.float32
                    )

                    embedding = np.asarray(
                        face.embedding,
                        dtype=np.float32
                    )

                    is_white, similarity = (
                        find_best_whitelist_match(
                            embedding,
                            whitelist
                        )
                    )

                    detections.append(
                        FaceDetection(
                            bbox=bbox,
                            embedding=embedding,
                            whitelisted=is_white,
                            similarity=similarity
                        )
                    )

                success = True

            except Exception as exc:

                print(
                    f"[AI] Inference error: {exc}"
                )

                # IMPORTANT:
                # Do not pretend this is a valid AI result.
                detections = []
                success = False

            latency_ms = (
                time.perf_counter()
                - inference_start
            ) * 1000.0

            timestamp = time.perf_counter()

            result = AIResult(
                seq_id=frame_seq,
                timestamp=timestamp,
                latency_ms=latency_ms,
                detections=detections,
                success=success
            )

            # ------------------------------------------------
            # Publish result
            # ------------------------------------------------

            with self._lock:
                if success:
                    self._latest_result = result

            self.last_latency_ms = latency_ms

            # ------------------------------------------------
            # AI FPS metrics
            # ------------------------------------------------

            self._result_count += 1

            elapsed = (
                time.perf_counter()
                - self._metrics_start
            )

            if elapsed >= 1.0:

                self.ai_fps = (
                    self._result_count / elapsed
                )

                self._result_count = 0
                self._metrics_start = time.perf_counter()


# ============================================================
# HYBRID TRACKER
# ============================================================

class HybridTracker:
    """
    Lightweight tracker running every output frame.

    AI is used to:
        - detect faces
        - identify the primary user
        - correct tracking drift
        - discover new faces

    Between AI results, this tracker uses motion prediction
    so the face position changes smoothly at the full video
    frame rate.
    """

    def __init__(self):

        self.active_tracks: Dict[int, Track] = {}

        self.next_track_id = 0

        self.last_ai_seq = 0

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self):
        self.active_tracks.clear()
        self.next_track_id = 0
        self.last_ai_seq = 0

    # --------------------------------------------------------
    # Main step
    # --------------------------------------------------------

    def step(
        self,
        ai_result: Optional[AIResult],
        current_time: float
    ):

        # ----------------------------------------------------
        # New AI result
        # ----------------------------------------------------

        if (
            ai_result is not None
            and ai_result.success
            and ai_result.seq_id != self.last_ai_seq
        ):

            self._correct_with_ai(
                ai_result,
                current_time
            )

            self.last_ai_seq = ai_result.seq_id

        # ----------------------------------------------------
        # Between AI results
        # ----------------------------------------------------

        else:

            self._predict()

        # ----------------------------------------------------
        # Identity timeout
        # ----------------------------------------------------

        self._apply_identity_timeouts(
            current_time
        )

    # --------------------------------------------------------
    # AI correction
    # --------------------------------------------------------

    def _correct_with_ai(
        self,
        ai_result: AIResult,
        current_time: float
    ):

        detections = ai_result.detections

        matched_tracks = set()
        matched_detections = set()

        # ----------------------------------------------------
        # Greedy spatial association
        # ----------------------------------------------------

        candidate_pairs = []

        for det_index, detection in enumerate(detections):

            det_center = get_centroid(
                detection.bbox
            )

            for track_id, track in self.active_tracks.items():

                if track_id in matched_tracks:
                    continue

                track_center = get_centroid(
                    track.bbox
                )

                distance = np.linalg.norm(
                    det_center - track_center
                )

                if distance <= MAX_MATCH_DISTANCE:

                    candidate_pairs.append(
                        (
                            distance,
                            det_index,
                            track_id
                        )
                    )

        # Closest matches first
        candidate_pairs.sort(
            key=lambda x: x[0]
        )

        for (
            distance,
            det_index,
            track_id
        ) in candidate_pairs:

            if det_index in matched_detections:
                continue

            if track_id in matched_tracks:
                continue

            detection = detections[
                det_index
            ]

            track = self.active_tracks[
                track_id
            ]

            # ------------------------------------------------
            # Estimate velocity
            # ------------------------------------------------

            old_center = get_centroid(
                track.bbox
            )

            new_center = get_centroid(
                detection.bbox
            )

            measured_velocity = (
                new_center - old_center
            )

            track.velocity = (
                track.velocity * (
                    1.0 - VELOCITY_SMOOTHING
                )
                +
                measured_velocity * VELOCITY_SMOOTHING
            )

            # ------------------------------------------------
            # Correct position
            # ------------------------------------------------

            track.bbox = (
                detection.bbox.copy()
            )

            track.lost_frames = 0

            track.confidence = min(
                1.0,
                track.confidence + 0.15
            )

            # ------------------------------------------------
            # Identity is ALWAYS determined by AI
            # ------------------------------------------------

            if detection.whitelisted:

                track.whitelisted = True

                track.last_verified_time = (
                    current_time
                )

            else:

                # IMPORTANT:
                # Never retain primary identity when AI
                # positively sees this track as a bystander.
                track.whitelisted = False

            matched_tracks.add(track_id)
            matched_detections.add(det_index)

        # ----------------------------------------------------
        # Create tracks for new faces
        # ----------------------------------------------------

        for det_index, detection in enumerate(detections):

            if det_index in matched_detections:
                continue

            track = Track(
                track_id=self.next_track_id,
                bbox=detection.bbox.copy(),
                velocity=np.zeros(
                    2,
                    dtype=np.float32
                ),
                whitelisted=detection.whitelisted,
                lost_frames=0,
                last_verified_time=(
                    current_time
                    if detection.whitelisted
                    else 0.0
                ),
                confidence=1.0
            )

            self.active_tracks[
                self.next_track_id
            ] = track

            self.next_track_id += 1

        # ----------------------------------------------------
        # Age unmatched tracks
        # ----------------------------------------------------

        for track_id, track in list(
            self.active_tracks.items()
        ):

            if track_id not in matched_tracks:

                track.lost_frames += 1

                # Keep track alive temporarily.
                # It will continue moving using its velocity.

                if (
                    track.lost_frames
                    > MAX_LOST_FRAMES
                ):
                    del self.active_tracks[
                        track_id
                    ]

    # --------------------------------------------------------
    # Predict movement
    # --------------------------------------------------------

    def _predict(self):

        for track in self.active_tracks.values():

            # Move bbox according to estimated velocity

            track.bbox[0] += track.velocity[0]
            track.bbox[2] += track.velocity[0]

            track.bbox[1] += track.velocity[1]
            track.bbox[3] += track.velocity[1]

            # Gradually reduce velocity.
            # This prevents runaway movement when AI has
            # not corrected the tracker for a while.

            track.velocity *= 0.92

            # Slowly reduce confidence

            track.confidence *= 0.995

    # --------------------------------------------------------
    # Identity timeout
    # --------------------------------------------------------

    def _apply_identity_timeouts(
        self,
        current_time: float
    ):

        for track in self.active_tracks.values():

            if not track.whitelisted:
                continue

            if track.last_verified_time <= 0:
                track.whitelisted = False
                continue

            verification_age = (
                current_time
                - track.last_verified_time
            )

            if (
                verification_age
                > PRIMARY_REVERIFY_TIMEOUT
            ):

                # Conservative privacy behavior
                track.whitelisted = False

    # --------------------------------------------------------
    # Find track at click position
    # --------------------------------------------------------

    def find_track_at(
        self,
        x: int,
        y: int
    ) -> Optional[Track]:

        for track in self.active_tracks.values():

            x1, y1, x2, y2 = track.bbox

            if (
                x1 <= x <= x2
                and
                y1 <= y <= y2
            ):
                return track

        return None


# ============================================================
# ENROLLMENT MANAGER
# ============================================================

class EnrollmentManager:

    def __init__(
        self,
        ai_worker: AsyncAIWorker,
        tracker: HybridTracker
    ):

        self.ai_worker = ai_worker
        self.tracker = tracker

        self.whitelist_embeddings: List[
            np.ndarray
        ] = []

        self.clicked_coords = None

    # --------------------------------------------------------
    # Mouse click
    # --------------------------------------------------------

    def set_click(
        self,
        x: int,
        y: int
    ):

        self.clicked_coords = (
            x,
            y
        )

    # --------------------------------------------------------
    # Process click
    # --------------------------------------------------------

    def process_click(
        self,
        ai_result: Optional[AIResult]
    ):

        if self.clicked_coords is None:
            return

        if ai_result is None:
            self.clicked_coords = None
            return

        x, y = self.clicked_coords

        track = self.tracker.find_track_at(
            x,
            y
        )

        if track is None:

            print(
                "[ENROLL] No tracked face at click."
            )

            self.clicked_coords = None
            return

        # ----------------------------------------------------
        # Find the newest AI detection belonging to track
        # ----------------------------------------------------

        best_detection = None
        best_distance = float("inf")

        track_center = get_centroid(
            track.bbox
        )

        for detection in ai_result.detections:

            detection_center = get_centroid(
                detection.bbox
            )

            distance = np.linalg.norm(
                track_center - detection_center
            )

            if distance < best_distance:

                best_distance = distance
                best_detection = detection

        if best_detection is None:

            print(
                "[ENROLL] Could not match AI detection."
            )

            self.clicked_coords = None
            return

        # ----------------------------------------------------
        # Enroll this face
        # ----------------------------------------------------

        self.whitelist_embeddings.append(
            best_detection.embedding.copy()
        )

        self.ai_worker.set_whitelist(
            self.whitelist_embeddings
        )

        # Make this track primary
        track.whitelisted = True

        track.last_verified_time = (
            time.perf_counter()
        )

        print(
            f"[ENROLL] Track {track.track_id} "
            f"registered as primary user."
        )

        self.clicked_coords = None

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self):

        self.whitelist_embeddings.clear()

        self.ai_worker.set_whitelist(
            self.whitelist_embeddings
        )

        self.tracker.reset()

        print(
            "[ENROLL] All permissions reset."
        )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

class PerformanceMetrics:

    def __init__(self):

        self.camera_fps = 0.0
        self.output_fps = 0.0

        self._camera_count = 0
        self._output_count = 0

        self._start_time = time.perf_counter()

    def tick_camera(self):

        self._camera_count += 1

    def tick_output(self):

        self._output_count += 1

        now = time.perf_counter()

        elapsed = (
            now - self._start_time
        )

        if elapsed >= 1.0:

            self.camera_fps = (
                self._camera_count / elapsed
            )

            self.output_fps = (
                self._output_count / elapsed
            )

            self._camera_count = 0
            self._output_count = 0

            self._start_time = now


# ============================================================
# INSIGHTFACE INITIALIZATION
# ============================================================

def initialize_insightface():

    print(
        "[INIT] Available ONNX providers:"
    )

    available = ort.get_available_providers()

    for provider in available:
        print(
            f"       {provider}"
        )

    # --------------------------------------------------------
    # Prefer CUDA when available
    # --------------------------------------------------------

    if "CUDAExecutionProvider" in available:

        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

        print(
            "[INIT] CUDA provider available."
        )

    else:

        providers = [
            "CPUExecutionProvider"
        ]

        print(
            "[INIT] CUDA unavailable. "
            "Using CPU."
        )

    # --------------------------------------------------------
    # Initialize InsightFace
    # --------------------------------------------------------

    face_app = FaceAnalysis(
        name=MODEL_NAME,
        providers=providers
    )

    face_app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    print(
        "[INIT] InsightFace initialized."
    )

    return face_app


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

def initialize_camera():

    print(
        f"[CAMERA] Requested resolution: "
        f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
    )

    print(
        f"[CAMERA] Requested FPS: {TARGET_FPS}"
    )

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera."
        )

    # --------------------------------------------------------
    # Reduce camera buffering
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    # --------------------------------------------------------
    # Request MJPG
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FPS,
        TARGET_FPS
    )

    # --------------------------------------------------------
    # Read actual values
    # --------------------------------------------------------

    actual_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    actual_fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or TARGET_FPS
    )

    print(
        f"[CAMERA] Actual mode: "
        f"{actual_width}x{actual_height} "
        f"@ {actual_fps:.1f} FPS"
    )

    return cap, actual_width, actual_height, actual_fps


# ============================================================
# MOUSE CALLBACK
# ============================================================

def create_mouse_callback(
    enrollment: EnrollmentManager
):

    def callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            enrollment.set_click(
                x,
                y
            )

    return callback


# ============================================================
# MAIN
# ============================================================

def main():

    global app

    print("=" * 60)
    print("PrivaStream - Hybrid Real-Time Privacy Pipeline")
    print("=" * 60)

    # --------------------------------------------------------
    # Initialize AI
    # --------------------------------------------------------

    app = initialize_insightface()

    # --------------------------------------------------------
    # Initialize camera
    # --------------------------------------------------------

    cap, width, height, camera_fps = (
        initialize_camera()
    )

    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    window_name = "PrivaStream"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    # --------------------------------------------------------
    # Create AI worker
    # --------------------------------------------------------

    ai_worker = AsyncAIWorker(
        app,
        AI_TARGET_FPS
    )

    ai_worker.start()

    # --------------------------------------------------------
    # Tracker
    # --------------------------------------------------------

    tracker = HybridTracker()

    # --------------------------------------------------------
    # Enrollment
    # --------------------------------------------------------

    enrollment = EnrollmentManager(
        ai_worker,
        tracker
    )

    cv2.setMouseCallback(
        window_name,
        create_mouse_callback(
            enrollment
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = PerformanceMetrics()

    # --------------------------------------------------------
    # Virtual camera
    # --------------------------------------------------------

    try:

        with pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=TARGET_FPS,
            fmt=pyvirtualcam.PixelFormat.BGR
        ) as vcam:

            print(
                f"[VCAM] Virtual camera running "
                f"at {width}x{height} @ {TARGET_FPS} FPS"
            )

            # Latest AI result seen by main loop
            latest_ai_result = None

            # ------------------------------------------------
            # Main real-time loop
            # ------------------------------------------------

            while True:

                # ============================================
                # 1. CAPTURE
                # ============================================

                ret, frame = cap.read()

                if not ret:

                    print(
                        "[CAMERA] Frame capture failed."
                    )

                    break

                metrics.tick_camera()

                current_time = (
                    time.perf_counter()
                )

                # ============================================
                # 2. SEND NEWEST FRAME TO AI
                # ============================================

                ai_worker.submit_frame(
                    frame
                )

                # ============================================
                # 3. GET LATEST AI RESULT
                # ============================================

                candidate_result = (
                    ai_worker.get_latest_result()
                )

                if candidate_result is not None:

                    if (
                        latest_ai_result is None
                        or
                        candidate_result.seq_id
                        >
                        latest_ai_result.seq_id
                    ):

                        latest_ai_result = (
                            candidate_result
                        )

                # ============================================
                # 4. CALCULATE AI AGE
                # ============================================

                if latest_ai_result is None:

                    ai_age = float("inf")

                else:

                    ai_age = (
                        current_time
                        -
                        latest_ai_result.timestamp
                    )

                # ============================================
                # 5. UPDATE TRACKER
                # ============================================

                tracker.step(
                    latest_ai_result,
                    current_time
                )

                # ============================================
                # 6. PROCESS ENROLLMENT
                # ============================================

                enrollment.process_click(
                    latest_ai_result
                )

                # ============================================
                # 7. FAIL-SAFE
                # ============================================

                failsafe_active = (
                    latest_ai_result is None
                    or
                    ai_age > FAILSAFE_TIMEOUT
                )

                # ============================================
                # 8. CREATE OUTPUT FRAME
                # ============================================

                output_frame = frame.copy()

                # ============================================
                # 9. APPLY PRIVACY BLUR
                # ============================================

                for track in tracker.active_tracks.values():

                    # Conservative privacy rule:
                    #
                    # Blur if:
                    #   - AI is stale
                    #   - face is not whitelisted
                    #
                    should_blur = (
                        failsafe_active
                        or
                        not track.whitelisted
                    )

                    if should_blur:

                        apply_fast_feathered_blur(
                            output_frame,
                            track.bbox
                        )

                # ============================================
                # 10. SEND TO VIRTUAL CAMERA
                # ============================================

                vcam.send(
                    output_frame
                )

                # ============================================
                # 11. CREATE LOCAL DISPLAY
                # ============================================

                display_frame = (
                    output_frame.copy()
                )

                # ============================================
                # 12. DRAW TRACKING HUD
                # ============================================

                for track in tracker.active_tracks.values():

                    x1, y1, x2, y2 = [
                        int(v)
                        for v in track.bbox
                    ]

                    # Keep rectangle inside image
                    x1 = max(
                        0,
                        min(width - 1, x1)
                    )

                    y1 = max(
                        0,
                        min(height - 1, y1)
                    )

                    x2 = max(
                        0,
                        min(width - 1, x2)
                    )

                    y2 = max(
                        0,
                        min(height - 1, y2)
                    )

                    if (
                        track.whitelisted
                        and
                        not failsafe_active
                    ):

                        status = "PRIMARY"
                        color = (
                            0,
                            255,
                            0
                        )

                    else:

                        status = "BLURRED"
                        color = (
                            0,
                            0,
                            255
                        )

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        display_frame,
                        f"ID:{track.track_id} "
                        f"[{status}]",
                        (
                            x1,
                            max(
                                20,
                                y1 - 8
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                # ============================================
                # 13. HUD
                # ============================================

                provider_text = "CPU"

                if (
                    "CUDAExecutionProvider"
                    in ort.get_available_providers()
                ):
                    provider_text = "CUDA/CPU"

                hud_lines = [

                    f"Output FPS: "
                    f"{metrics.output_fps:.1f}",

                    f"Camera FPS: "
                    f"{metrics.camera_fps:.1f}",

                    f"AI FPS: "
                    f"{ai_worker.ai_fps:.1f}",

                    f"AI Latency: "
                    f"{ai_worker.last_latency_ms:.1f} ms",

                    f"AI Age: "
                    f"{ai_age * 1000:.0f} ms",

                    f"AI Provider: "
                    f"{provider_text}",

                    f"Resolution: "
                    f"{width}x{height}",

                    f"Tracks: "
                    f"{len(tracker.active_tracks)}",

                    (
                        "FAIL-SAFE ACTIVE"
                        if failsafe_active
                        else "PRIVACY ACTIVE"
                    )
                ]

                y = 25

                for line in hud_lines:

                    cv2.putText(
                        display_frame,
                        line,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )

                    y += 22

                # ============================================
                # 14. PERFORMANCE METRICS
                # ============================================

                metrics.tick_output()

                # ============================================
                # 15. DISPLAY
                # ============================================

                cv2.imshow(
                    window_name,
                    display_frame
                )

                # ============================================
                # 16. KEYBOARD
                # ============================================

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                # Q = quit
                if key == ord("q"):

                    break

                # R = reset enrollment
                elif key == ord("r"):

                    enrollment.reset()

    finally:

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        ai_worker.stop()

        cap.release()

        cv2.destroyAllWindows()

        print(
            "[SYSTEM] PrivaStream stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()