import cv2
import numpy as np
import pyvirtualcam
import threading
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import onnxruntime as ort


# ============================================================
# PrivaStream - SAMPLE 5
# Hybrid 60 FPS Output + Asynchronous GPU AI
# Supports:
#   1. Live webcam
#   2. Prerecorded video
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "540p": (960, 540),
    "360p": (640, 360),
}

ACTIVE_PRESET = "720p"

# ============================================================
# INPUT SOURCE
#
# Webcam:
#     VIDEO_SOURCE = 0
#
# Video file:
#     VIDEO_SOURCE = "test.mp4"
#
# You can also use a full path:
#     VIDEO_SOURCE = r"C:\Users\YourName\Videos\test.mp4"
# ============================================================

VIDEO_SOURCE = 0

CAMERA_WIDTH, CAMERA_HEIGHT = RESOLUTION_PRESETS[ACTIVE_PRESET]

# Output FPS.
# For webcam this is the target FPS.
# For prerecorded video, the video's FPS will be used when possible.
TARGET_FPS = 60.0

AI_TARGET_FPS = 3.0

MODEL_NAME = "buffalo_s"

SIMILARITY_THRESHOLD = 0.44

MAX_MATCH_DISTANCE = 180.0

REID_THRESHOLD = 0.48

MAX_LOST_FRAMES = 45

MAX_PREDICT_FRAMES = 12

VELOCITY_SMOOTHING = 0.35

PRIMARY_REVERIFY_TIMEOUT = 1.5

FAILSAFE_TIMEOUT = 0.75

BLUR_DOWNSAMPLE = 4

CAMERA_BUFFER_SIZE = 1


# ============================================================
# 2. CUDA / ONNX RUNTIME DLL PRELOAD
# ============================================================

print("[INIT] Preloading CUDA/cuDNN DLLs...")

try:

    ort.preload_dlls(directory="")

    print(
        "[INIT] CUDA/cuDNN DLL preload completed."
    )

except Exception as e:

    print(
        f"[INIT] CUDA DLL preload warning: {e}"
    )


from insightface.app import FaceAnalysis


# ============================================================
# 3. DATA STRUCTURES
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

    detections: List[FaceDetection] = field(
        default_factory=list
    )


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


# ============================================================
# 4. COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
        +
        1e-8
    )

    return float(
        np.dot(a, b)
        /
        denominator
    )


def check_identity(
    embedding: np.ndarray,
    whitelist: List[np.ndarray],
    threshold: float
) -> Tuple[bool, float]:

    if not whitelist:

        return False, 0.0

    best_similarity = -1.0

    for saved in whitelist:

        similarity = cosine_similarity(
            embedding,
            saved
        )

        if similarity > best_similarity:

            best_similarity = similarity

    return (
        best_similarity >= threshold,
        best_similarity
    )


# ============================================================
# 5. INSIGHTFACE INITIALIZATION
# ============================================================

def initialize_insightface():

    print(
        f"[INIT] Loading {MODEL_NAME} "
        f"with GPU/CPU provider detection..."
    )

    available = ort.get_available_providers()

    print(
        "[INIT] ONNX Runtime providers:",
        available
    )

    if "CUDAExecutionProvider" in available:

        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

        print(
            "[INIT] Using NVIDIA CUDA GPU "
            "for InsightFace."
        )

    else:

        providers = [
            "CPUExecutionProvider"
        ]

        print(
            "[INIT] CUDAExecutionProvider unavailable. "
            "Using CPU."
        )

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=providers
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    active_provider = "CPUExecutionProvider"

    try:

        for model in app.models.values():

            if hasattr(model, "session"):

                session_providers = (
                    model.session.get_providers()
                )

                if session_providers:

                    active_provider = (
                        session_providers[0]
                    )

                    break

    except Exception:
        pass

    print(
        f"[INIT] InsightFace active provider: "
        f"{active_provider}"
    )

    if active_provider != "CUDAExecutionProvider":

        print(
            "[WARNING] InsightFace is NOT using CUDA."
        )

    return app, active_provider


# ============================================================
# 6. ASYNCHRONOUS AI WORKER
# ============================================================

class AsyncAIWorker:

    def __init__(
        self,
        app: FaceAnalysis,
        target_fps: float = 8.0
    ):

        self.app = app

        self.target_interval = (
            1.0 /
            max(
                1.0,
                target_fps
            )
        )

        self.running = False

        self.thread: Optional[
            threading.Thread
        ] = None

        self.lock = threading.Lock()

        self.latest_frame: Optional[
            np.ndarray
        ] = None

        self.whitelist_embeddings: List[
            np.ndarray
        ] = []

        self.threshold = SIMILARITY_THRESHOLD

        self.latest_result = AIResult(
            seq_id=0,
            timestamp=0.0,
            latency_ms=0.0,
            detections=[]
        )

        self.sequence = 0

        self.fps = 0.0

        self.latency_ms = 0.0

        self._count = 0

        self._fps_timer = time.perf_counter()

    def start(self):

        if self.running:

            return

        self.running = True

        self.thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="PrivaStream-AI"
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if (
            self.thread is not None
            and
            self.thread.is_alive()
        ):

            self.thread.join(
                timeout=2.0
            )

    def submit_frame(
        self,
        frame: np.ndarray
    ):

        with self.lock:

            self.latest_frame = frame.copy()

    def update_whitelist(
        self,
        embeddings: List[np.ndarray]
    ):

        with self.lock:

            self.whitelist_embeddings = [
                e.copy()
                for e in embeddings
            ]

    def get_latest_result(self) -> AIResult:

        with self.lock:

            return self.latest_result

    def _worker_loop(self):

        while self.running:

            loop_start = time.perf_counter()

            frame = None

            with self.lock:

                if self.latest_frame is not None:

                    frame = self.latest_frame

                    self.latest_frame = None

                whitelist = [
                    e.copy()
                    for e in
                    self.whitelist_embeddings
                ]

                threshold = self.threshold

            if frame is None:

                time.sleep(0.002)

                continue

            infer_start = time.perf_counter()

            detections = []

            try:

                faces = self.app.get(
                    frame
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

                    is_primary, similarity = (
                        check_identity(
                            embedding,
                            whitelist,
                            threshold
                        )
                    )

                    detections.append(
                        FaceDetection(
                            bbox=bbox,
                            embedding=embedding,
                            whitelisted=is_primary,
                            similarity=similarity
                        )
                    )

            except Exception as e:

                print(
                    f"[AI WORKER ERROR] {e}"
                )

            latency_ms = (
                time.perf_counter()
                -
                infer_start
            ) * 1000.0

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

            now = time.perf_counter()

            elapsed = (
                now -
                self._fps_timer
            )

            if elapsed >= 1.0:

                self.fps = (
                    self._count /
                    elapsed
                )

                self._count = 0

                self._fps_timer = now

            work_time = (
                time.perf_counter()
                -
                loop_start
            )

            remaining = (
                self.target_interval -
                work_time
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )


# ============================================================
# 7. HYBRID TRACKER
# ============================================================

class HybridTracker:

    def __init__(self):

        self.active_tracks: Dict[
            int,
            Track
        ] = {}

        self.next_track_id = 0

        self.last_ai_seq = -1

    @staticmethod
    def center(
        bbox: np.ndarray
    ) -> np.ndarray:

        return np.array(
            [
                (bbox[0] + bbox[2]) * 0.5,
                (bbox[1] + bbox[3]) * 0.5
            ],
            dtype=np.float32
        )

    @staticmethod
    def iou(
        a: np.ndarray,
        b: np.ndarray
    ) -> float:

        x1 = max(
            a[0],
            b[0]
        )

        y1 = max(
            a[1],
            b[1]
        )

        x2 = min(
            a[2],
            b[2]
        )

        y2 = min(
            a[3],
            b[3]
        )

        iw = max(
            0.0,
            x2 - x1
        )

        ih = max(
            0.0,
            y2 - y1
        )

        intersection = iw * ih

        if intersection <= 0:

            return 0.0

        area_a = (
            max(
                0.0,
                a[2] - a[0]
            )
            *
            max(
                0.0,
                a[3] - a[1]
            )
        )

        area_b = (
            max(
                0.0,
                b[2] - b[0]
            )
            *
            max(
                0.0,
                b[3] - b[1]
            )
        )

        union = (
            area_a +
            area_b -
            intersection
        )

        if union <= 0:

            return 0.0

        return float(
            intersection /
            union
        )

    def step(
        self,
        ai_result: AIResult,
        now: float
    ):

        if (
            ai_result.seq_id >
            self.last_ai_seq
        ):

            self._apply_ai_result(
                ai_result,
                now
            )

            self.last_ai_seq = (
                ai_result.seq_id
            )

        else:

            self._predict_one_frame(
                now
            )

    def _apply_ai_result(
        self,
        result: AIResult,
        now: float
    ):

        detections = result.detections

        matched_tracks = set()

        matched_detections = set()

        candidates = []

        for di, detection in enumerate(
            detections
        ):

            det_center = self.center(
                detection.bbox
            )

            for track_id, track in (
                self.active_tracks.items()
            ):

                if track_id in matched_tracks:

                    continue

                track_center = self.center(
                    track.bbox
                )

                distance = float(
                    np.linalg.norm(
                        det_center -
                        track_center
                    )
                )

                embedding_similarity = (
                    cosine_similarity(
                        detection.embedding,
                        track.embedding
                    )
                )

                spatial_ok = (
                    distance <=
                    MAX_MATCH_DISTANCE
                )

                reid_ok = (
                    embedding_similarity >=
                    REID_THRESHOLD
                )

                if (
                    not spatial_ok
                    and
                    not reid_ok
                ):

                    continue

                embedding_cost = (
                    1.0 -
                    max(
                        -1.0,
                        min(
                            1.0,
                            embedding_similarity
                        )
                    )
                )

                normalized_distance = min(
                    distance /
                    max(
                        1.0,
                        MAX_MATCH_DISTANCE
                    ),
                    2.0
                )

                score = (
                    embedding_cost * 2.0
                    +
                    normalized_distance
                )

                if spatial_ok:

                    score -= 0.20

                candidates.append(
                    (
                        score,
                        embedding_similarity,
                        distance,
                        di,
                        track_id
                    )
                )

        candidates.sort(
            key=lambda x: x[0]
        )

        for (
            score,
            embedding_similarity,
            distance,
            di,
            track_id
        ) in candidates:

            if di in matched_detections:

                continue

            if track_id in matched_tracks:

                continue

            detection = detections[di]

            track = self.active_tracks[
                track_id
            ]

            old_center = self.center(
                track.bbox
            )

            new_center = self.center(
                detection.bbox
            )

            movement = (
                new_center -
                old_center
            )

            elapsed_frames = max(
                1.0,
                track.lost_frames +
                1.0
            )

            measured_velocity = (
                movement /
                elapsed_frames
            )

            track.velocity = (
                (
                    1.0 -
                    VELOCITY_SMOOTHING
                )
                *
                track.velocity
                +
                VELOCITY_SMOOTHING
                *
                measured_velocity
            )

            track.bbox = (
                detection.bbox.copy()
            )

            track.embedding = (
                0.70 *
                track.embedding
                +
                0.30 *
                detection.embedding
            )

            norm = np.linalg.norm(
                track.embedding
            )

            if norm > 1e-8:

                track.embedding /= norm

            track.lost_frames = 0

            track.predict_frames = 0

            track.confidence = (
                detection.similarity
            )

            track.last_update_time = now

            if detection.whitelisted:

                track.whitelisted = True

                track.last_verified_time = now

            else:

                track.whitelisted = False

            matched_tracks.add(
                track_id
            )

            matched_detections.add(
                di
            )

        for di, detection in enumerate(
            detections
        ):

            if di in matched_detections:

                continue

            new_id = self.next_track_id

            self.next_track_id += 1

            self.active_tracks[
                new_id
            ] = Track(
                track_id=new_id,
                bbox=detection.bbox.copy(),
                embedding=detection.embedding.copy(),
                velocity=np.zeros(
                    2,
                    dtype=np.float32
                ),
                whitelisted=detection.whitelisted,
                confidence=detection.similarity,
                lost_frames=0,
                predict_frames=0,
                last_verified_time=(
                    now
                    if detection.whitelisted
                    else 0.0
                ),
                last_update_time=now
            )

            matched_tracks.add(
                new_id
            )

        dead = []

        for track_id, track in (
            self.active_tracks.items()
        ):

            if track_id in matched_tracks:

                continue

            track.lost_frames += 1

            track.predict_frames += 1

            if (
                track.whitelisted
                and
                (
                    now -
                    track.last_verified_time
                    >
                    PRIMARY_REVERIFY_TIMEOUT
                )
            ):

                track.whitelisted = False

            if (
                track.predict_frames
                <=
                MAX_PREDICT_FRAMES
            ):

                track.bbox[0] += (
                    track.velocity[0]
                )

                track.bbox[2] += (
                    track.velocity[0]
                )

                track.bbox[1] += (
                    track.velocity[1]
                )

                track.bbox[3] += (
                    track.velocity[1]
                )

                track.velocity *= 0.92

            else:

                track.velocity *= 0.50

            if (
                track.lost_frames
                >
                MAX_LOST_FRAMES
            ):

                dead.append(
                    track_id
                )

        for track_id in dead:

            del self.active_tracks[
                track_id
            ]

    def _predict_one_frame(
        self,
        now: float
    ):

        dead = []

        for track_id, track in (
            self.active_tracks.items()
        ):

            track.lost_frames += 1

            track.predict_frames += 1

            if (
                track.whitelisted
                and
                (
                    now -
                    track.last_verified_time
                    >
                    PRIMARY_REVERIFY_TIMEOUT
                )
            ):

                track.whitelisted = False

            if (
                track.predict_frames
                <=
                MAX_PREDICT_FRAMES
            ):

                track.bbox[0] += (
                    track.velocity[0]
                )

                track.bbox[2] += (
                    track.velocity[0]
                )

                track.bbox[1] += (
                    track.velocity[1]
                )

                track.bbox[3] += (
                    track.velocity[1]
                )

                track.velocity *= 0.98

            if (
                track.lost_frames
                >
                MAX_LOST_FRAMES
            ):

                dead.append(
                    track_id
                )

        for track_id in dead:

            del self.active_tracks[
                track_id
            ]

    def reset(self):

        self.active_tracks.clear()

        self.next_track_id = 0

        self.last_ai_seq = -1


# ============================================================
# 8. FAST FEATHERED BLUR
# ============================================================

def apply_fast_blur(
    frame: np.ndarray,
    bbox: np.ndarray
):

    h, w = frame.shape[:2]

    x1 = max(
        0,
        min(
            w - 1,
            int(bbox[0])
        )
    )

    y1 = max(
        0,
        min(
            h - 1,
            int(bbox[1])
        )
    )

    x2 = max(
        0,
        min(
            w,
            int(bbox[2])
        )
    )

    y2 = max(
        0,
        min(
            h,
            int(bbox[3])
        )
    )

    if x2 <= x1 or y2 <= y1:

        return

    roi = frame[
        y1:y2,
        x1:x2
    ]

    roi_h, roi_w = roi.shape[:2]

    if roi_h < 8 or roi_w < 8:

        return

    small_w = max(
        4,
        roi_w //
        BLUR_DOWNSAMPLE
    )

    small_h = max(
        4,
        roi_h //
        BLUR_DOWNSAMPLE
    )

    small = cv2.resize(
        roi,
        (
            small_w,
            small_h
        ),
        interpolation=cv2.INTER_LINEAR
    )

    blurred_small = cv2.GaussianBlur(
        small,
        (15, 15),
        0
    )

    blurred = cv2.resize(
        blurred_small,
        (
            roi_w,
            roi_h
        ),
        interpolation=cv2.INTER_LINEAR
    )

    mask = np.zeros(
        (
            roi_h,
            roi_w
        ),
        dtype=np.uint8
    )

    center = (
        roi_w // 2,
        roi_h // 2
    )

    axes = (
        max(
            1,
            int(
                roi_w *
                0.48
            )
        ),
        max(
            1,
            int(
                roi_h *
                0.48
            )
        )
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

    feather = max(
        5,
        (
            min(
                roi_w,
                roi_h
            )
            //
            6
        )
        | 1
    )

    feather = min(
        feather,
        31
    )

    mask = cv2.GaussianBlur(
        mask,
        (
            feather,
            feather
        ),
        0
    )

    alpha = (
        mask.astype(
            np.float32
        )
        /
        255.0
    )

    alpha = alpha[
        ...,
        None
    ]

    blended = (
        roi.astype(
            np.float32
        )
        *
        (1.0 - alpha)
        +
        blurred.astype(
            np.float32
        )
        *
        alpha
    )

    roi[:] = np.clip(
        blended,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# 9. ENROLLMENT MANAGER
# ============================================================

class EnrollmentManager:

    def __init__(self):

        self.whitelist_embeddings: List[
            np.ndarray
        ] = []

        self.clicked_coords: Optional[
            Tuple[int, int]
        ] = None

        self.lock = threading.Lock()

    def on_mouse_click(
        self,
        event,
        x,
        y,
        flags,
        param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            with self.lock:

                self.clicked_coords = (
                    x,
                    y
                )

    def process_click(
        self,
        tracker: HybridTracker,
        ai_result: AIResult,
        ai_worker: AsyncAIWorker
    ):

        with self.lock:

            coords = self.clicked_coords

            self.clicked_coords = None

        if coords is None:

            return

        cx, cy = coords

        selected_detection = None

        for detection in ai_result.detections:

            x1, y1, x2, y2 = (
                detection.bbox
            )

            if (
                x1 <= cx <= x2
                and
                y1 <= cy <= y2
            ):

                selected_detection = (
                    detection
                )

                break

        if selected_detection is None:

            print(
                "[ENROLL] No fresh AI face found "
                "at clicked position. Try again."
            )

            return

        self.whitelist_embeddings.append(
            selected_detection.embedding.copy()
        )

        ai_worker.update_whitelist(
            self.whitelist_embeddings
        )

        clicked_center = np.array(
            [
                cx,
                cy
            ],
            dtype=np.float32
        )

        best_track = None

        best_distance = float("inf")

        for track in (
            tracker.active_tracks.values()
        ):

            center = tracker.center(
                track.bbox
            )

            distance = np.linalg.norm(
                clicked_center -
                center
            )

            if distance < best_distance:

                best_distance = distance

                best_track = track

        if best_track is not None:

            best_track.whitelisted = True

            best_track.last_verified_time = (
                time.time()
            )

            best_track.confidence = (
                selected_detection.similarity
            )

            best_track.embedding = (
                selected_detection.embedding.copy()
            )

            print(
                f"[ENROLL] Track "
                f"{best_track.track_id} "
                f"registered as PRIMARY."
            )

        else:

            print(
                "[ENROLL] Primary embedding saved."
            )

    def reset(
        self,
        tracker: HybridTracker,
        ai_worker: AsyncAIWorker
    ):

        self.whitelist_embeddings.clear()

        ai_worker.update_whitelist(
            self.whitelist_embeddings
        )

        tracker.reset()

        print(
            "[ENROLL] All primary identities reset."
        )


# ============================================================
# 10. PERFORMANCE METRICS
# ============================================================

class PerformanceMetrics:

    def __init__(self):

        self.camera_fps = 0.0

        self.output_fps = 0.0

        self.main_loop_fps = 0.0

        self._camera_count = 0

        self._output_count = 0

        self._loop_count = 0

        now = time.perf_counter()

        self._camera_timer = now

        self._output_timer = now

        self._loop_timer = now

    def tick_camera(self):

        self._camera_count += 1

        now = time.perf_counter()

        elapsed = (
            now -
            self._camera_timer
        )

        if elapsed >= 1.0:

            self.camera_fps = (
                self._camera_count /
                elapsed
            )

            self._camera_count = 0

            self._camera_timer = now

    def tick_output(self):

        self._output_count += 1

        now = time.perf_counter()

        elapsed = (
            now -
            self._output_timer
        )

        if elapsed >= 1.0:

            self.output_fps = (
                self._output_count /
                elapsed
            )

            self._output_count = 0

            self._output_timer = now

    def tick_loop(self):

        self._loop_count += 1

        now = time.perf_counter()

        elapsed = (
            now -
            self._loop_timer
        )

        if elapsed >= 1.0:

            self.main_loop_fps = (
                self._loop_count /
                elapsed
            )

            self._loop_count = 0

            self._loop_timer = now


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PrivaStream - SAMPLE 5"
    )

    print(
        "Hybrid 60 FPS Output + Async GPU AI"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Detect input type
    # --------------------------------------------------------

    is_webcam = isinstance(
        VIDEO_SOURCE,
        int
    )

    if is_webcam:

        print(
            f"[INPUT] Source: WEBCAM {VIDEO_SOURCE}"
        )

    else:

        print(
            f"[INPUT] Source: VIDEO FILE"
        )

        print(
            f"[INPUT] File: {VIDEO_SOURCE}"
        )

    # --------------------------------------------------------
    # InsightFace
    # --------------------------------------------------------

    app, active_provider = (
        initialize_insightface()
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    print(
        "[INPUT] Opening source..."
    )

    if is_webcam:

        cap = cv2.VideoCapture(
            VIDEO_SOURCE,
            cv2.CAP_DSHOW
        )

        if not cap.isOpened():

            raise RuntimeError(
                "Could not open webcam."
            )

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            CAMERA_BUFFER_SIZE
        )

        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(
                *"MJPG"
            )
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

        source_fps = (
            cap.get(
                cv2.CAP_PROP_FPS
            )
            or
            TARGET_FPS
        )

        output_fps = TARGET_FPS

    else:

        cap = cv2.VideoCapture(
            VIDEO_SOURCE
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Could not open video file:\n"
                f"{VIDEO_SOURCE}"
            )

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

        source_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if (
            source_fps <= 0
            or
            np.isnan(source_fps)
        ):

            source_fps = TARGET_FPS

        # For prerecorded videos, preserve the video's
        # original FPS rather than forcing 60 FPS.
        output_fps = source_fps

    print(
        f"[INPUT] "
        f"{actual_width}x{actual_height} "
        f"@ {source_fps:.2f} FPS"
    )

    print(
        f"[OUTPUT] Virtual camera FPS: "
        f"{output_fps:.2f}"
    )

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    ai_worker = AsyncAIWorker(
        app,
        AI_TARGET_FPS
    )

    tracker = HybridTracker()

    enrollment = EnrollmentManager()

    metrics = PerformanceMetrics()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    window_name = (
        "PrivaStream - Sample 5"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        actual_width,
        actual_height
    )

    cv2.setMouseCallback(
        window_name,
        enrollment.on_mouse_click
    )

    # --------------------------------------------------------
    # Virtual camera
    # --------------------------------------------------------

    vcam = None

    try:

        vcam = pyvirtualcam.Camera(
            width=actual_width,
            height=actual_height,
            fps=output_fps,
            fmt=pyvirtualcam.PixelFormat.BGR
        )

        print(
            f"[VCAM] Virtual camera started: "
            f"{vcam.device}"
        )

    except Exception as e:

        cap.release()

        cv2.destroyAllWindows()

        raise RuntimeError(
            f"Could not start virtual camera: {e}"
        )

    # --------------------------------------------------------
    # Start AI
    # --------------------------------------------------------

    ai_worker.start()

    print(
        f"[AI] Background InsightFace worker "
        f"started at target "
        f"{AI_TARGET_FPS:.1f} FPS."
    )

    print()

    print(
        "[CONTROLS]"
    )

    print(
        "  Click a face  -> enroll as PRIMARY"
    )

    print(
        "  R             -> reset identities"
    )

    print(
        "  Q             -> quit"
    )

    print()

    # --------------------------------------------------------
    # Main video loop
    # --------------------------------------------------------

    try:

        while True:

            loop_start = (
                time.perf_counter()
            )

            # =================================================
            # A. CAPTURE
            # =================================================

            ret, frame = cap.read()

            if not ret:

                if is_webcam:

                    print(
                        "[CAMERA] Frame capture failed."
                    )

                else:

                    print(
                        "[VIDEO] Video finished."
                    )

                break

            # -------------------------------------------------
            # Make sure frame matches expected output size.
            # -------------------------------------------------

            if (
                frame.shape[1] != actual_width
                or
                frame.shape[0] != actual_height
            ):

                frame = cv2.resize(
                    frame,
                    (
                        actual_width,
                        actual_height
                    ),
                    interpolation=cv2.INTER_LINEAR
                )

            metrics.tick_camera()

            # =================================================
            # B. SUBMIT FRAME TO AI
            # =================================================

            ai_worker.submit_frame(
                frame
            )

            # =================================================
            # C. GET LATEST AI RESULT
            # =================================================

            ai_result = (
                ai_worker.get_latest_result()
            )

            now = time.time()

            if ai_result.timestamp > 0:

                ai_age = (
                    now -
                    ai_result.timestamp
                )

            else:

                ai_age = float("inf")

            # =================================================
            # D. TRACK
            # =================================================

            tracker_start = (
                time.perf_counter()
            )

            tracker.step(
                ai_result,
                now
            )

            tracking_ms = (
                time.perf_counter()
                -
                tracker_start
            ) * 1000.0

            # =================================================
            # E. ENROLLMENT
            # =================================================

            enrollment.process_click(
                tracker,
                ai_result,
                ai_worker
            )

            # =================================================
            # F. FAILSAFE
            # =================================================

            failsafe_active = (
                ai_result.seq_id == 0
                or
                ai_age > FAILSAFE_TIMEOUT
            )

            # =================================================
            # G. PRIVACY RENDER
            # =================================================

            output_frame = frame.copy()

            blur_start = (
                time.perf_counter()
            )

            for track in list(
                tracker.active_tracks.values()
            ):

                if (
                    failsafe_active
                    or
                    not track.whitelisted
                ):

                    apply_fast_blur(
                        output_frame,
                        track.bbox
                    )

            blur_ms = (
                time.perf_counter()
                -
                blur_start
            ) * 1000.0

            # =================================================
            # H. SEND TO VIRTUAL CAMERA
            # =================================================

            vcam.send(
                output_frame
            )

            metrics.tick_output()

            # =================================================
            # I. LOCAL PREVIEW
            # =================================================

            display_frame = (
                output_frame.copy()
            )

            for track in (
                tracker.active_tracks.values()
            ):

                x1, y1, x2, y2 = [
                    int(v)
                    for v in track.bbox
                ]

                x1 = max(
                    0,
                    min(
                        actual_width - 1,
                        x1
                    )
                )

                y1 = max(
                    0,
                    min(
                        actual_height - 1,
                        y1
                    )
                )

                x2 = max(
                    0,
                    min(
                        actual_width - 1,
                        x2
                    )
                )

                y2 = max(
                    0,
                    min(
                        actual_height - 1,
                        y2
                    )
                )

                if failsafe_active:

                    color = (
                        0,
                        165,
                        255
                    )

                    status = (
                        "FAIL-SAFE"
                    )

                elif track.whitelisted:

                    color = (
                        0,
                        255,
                        0
                    )

                    status = (
                        f"PRIMARY "
                        f"{track.confidence:.2f}"
                    )

                else:

                    color = (
                        0,
                        0,
                        255
                    )

                    status = (
                        "BLURRED"
                    )

                cv2.rectangle(
                    display_frame,
                    (
                        x1,
                        y1
                    ),
                    (
                        x2,
                        y2
                    ),
                    color,
                    2
                )

                cv2.putText(
                    display_frame,
                    (
                        f"ID:{track.track_id} "
                        f"[{status}]"
                    ),
                    (
                        x1,
                        max(
                            22,
                            y1 - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2
                )

            # =================================================
            # J. HUD
            # =================================================

            metrics.tick_loop()

            mode_text = (
                "WEBCAM"
                if is_webcam
                else "VIDEO"
            )

            hud_lines = [

                (
                    f"OUTPUT: "
                    f"{metrics.output_fps:.1f} FPS"
                ),

                (
                    f"INPUT: "
                    f"{metrics.camera_fps:.1f} FPS"
                ),

                (
                    f"AI: "
                    f"{ai_worker.fps:.1f} FPS "
                    f"({ai_worker.latency_ms:.1f} ms)"
                ),

                (
                    f"TRACK: "
                    f"{tracking_ms:.2f} ms"
                ),

                (
                    f"BLUR: "
                    f"{blur_ms:.2f} ms"
                ),

                (
                    f"FACES: "
                    f"{len(tracker.active_tracks)}"
                ),

                (
                    f"ENGINE: "
                    f"{'CUDA' if active_provider == 'CUDAExecutionProvider' else 'CPU'}"
                ),

                (
                    f"MODE: "
                    f"{mode_text} / ASYNC AI"
                ),

                (
                    "STATE: "
                    +
                    (
                        "FAIL-SAFE"
                        if failsafe_active
                        else "PRIVACY ACTIVE"
                    )
                ),
            ]

            hud_height = (
                18 +
                len(hud_lines) * 23
            )

            cv2.rectangle(
                display_frame,
                (
                    10,
                    10
                ),
                (
                    350,
                    hud_height
                ),
                (0, 0, 0),
                -1
            )

            y = 31

            for text in hud_lines:

                if text.startswith(
                    "STATE:"
                ):

                    hud_color = (
                        0,
                        165,
                        255
                    ) if failsafe_active else (
                        0,
                        255,
                        0
                    )

                    thickness = 2

                elif text.startswith(
                    "OUTPUT:"
                ):

                    hud_color = (
                        255,
                        255,
                        0
                    )

                    thickness = 2

                else:

                    hud_color = (
                        220,
                        220,
                        220
                    )

                    thickness = 1

                cv2.putText(
                    display_frame,
                    text,
                    (
                        20,
                        y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    hud_color,
                    thickness
                )

                y += 23

            # =================================================
            # K. DISPLAY
            # =================================================

            cv2.imshow(
                window_name,
                display_frame
            )

            # =================================================
            # L. KEYBOARD
            # =================================================

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == ord("q"):

                print(
                    "[SYSTEM] Quit requested."
                )

                break

            elif key == ord("r"):

                enrollment.reset(
                    tracker,
                    ai_worker
                )

            # =================================================
            # M. OUTPUT PACING
            # =================================================

            vcam.sleep_until_next_frame()

    except KeyboardInterrupt:

        print(
            "\n[SYSTEM] Keyboard interrupt."
        )

    finally:

        print(
            "[SYSTEM] Cleaning up..."
        )

        ai_worker.stop()

        if vcam is not None:

            try:

                vcam.close()

            except Exception:

                pass

        cap.release()

        cv2.destroyAllWindows()

        print(
            "[SYSTEM] PrivaStream stopped cleanly."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()