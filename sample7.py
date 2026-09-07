import cv2
import os
import numpy as np
import pyvirtualcam
import threading
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import onnxruntime as ort


# ============================================================
# PrivaStream - SAMPLE 6
# Hybrid 60 FPS Output + Asynchronous GPU AI
#
# Architecture:
#   Camera            -> main thread, every frame
#   InsightFace       -> background worker, ~8 FPS
#   Tracker           -> main thread, every frame
#   Emoji mask        -> main thread, every frame
#   Virtual camera    -> main thread, target 60 FPS
#
# AI is never allowed to block the video/output loop.
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

CAMERA_WIDTH, CAMERA_HEIGHT = RESOLUTION_PRESETS[ACTIVE_PRESET]

TARGET_FPS = 60.0

# GPU benchmark showed buffalo_s running around 7-14 AI FPS.
# 8 FPS is therefore a reasonable target.
AI_TARGET_FPS = 3.0

MODEL_NAME = "buffalo_s"

SIMILARITY_THRESHOLD = 0.44

# Track/detection association.
MAX_MATCH_DISTANCE = 180.0

# Re-identification threshold between an old track embedding
# and a newly detected face.
REID_THRESHOLD = 0.48

# How long a lost track is retained.
# At 60 FPS, 45 frames ~= 0.75 seconds.
MAX_LOST_FRAMES = 45

# For a short period after losing a face, use velocity prediction.
# After this, stop extrapolating so the blur box does not "fly away".
MAX_PREDICT_FRAMES = 12

VELOCITY_SMOOTHING = 0.35

# Identity safety:
# A PRIMARY track must periodically be positively verified by AI.
PRIMARY_REVERIFY_TIMEOUT = 1.5

# AI result considered stale after this amount of time.
FAILSAFE_TIMEOUT = 0.75

# Emoji mask.
BLUR_DOWNSAMPLE = 4

# Camera buffer.
CAMERA_BUFFER_SIZE = 1


# ============================================================
# 2. CUDA / ONNX RUNTIME DLL PRELOAD
# ============================================================

print("[INIT] Preloading CUDA/cuDNN DLLs...")

try:
    ort.preload_dlls(directory="")
    print("[INIT] CUDA/cuDNN DLL preload completed.")
except Exception as e:
    print(f"[INIT] CUDA DLL preload warning: {e}")


# InsightFace is imported AFTER the CUDA DLL preload.
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

    # Manual privacy override.
    # True = temporarily reveal this bystander.
    manual_revealed: bool = False


# ============================================================
# 4. COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
        + 1e-8
    )

    return float(
        np.dot(a, b) / denominator
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
        f"[INIT] Loading {MODEL_NAME} with GPU/CPU provider detection..."
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
            "[INIT] Using NVIDIA CUDA GPU for InsightFace."
        )

    else:

        providers = [
            "CPUExecutionProvider"
        ]

        print(
            "[INIT] CUDAExecutionProvider unavailable."
            " Using CPU."
        )

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=providers
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    # Verify the actual provider used by at least one model.
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
            1.0 / max(1.0, target_fps)
        )

        self.running = False

        self.thread: Optional[
            threading.Thread
        ] = None

        self.lock = threading.Lock()

        # Latest-frame-wins input buffer.
        self.latest_frame: Optional[
            np.ndarray
        ] = None

        # Whitelist copied safely into worker.
        self.whitelist_embeddings: List[
            np.ndarray
        ] = []

        self.threshold = SIMILARITY_THRESHOLD

        # Latest result.
        self.latest_result = AIResult(
            seq_id=0,
            timestamp=0.0,
            latency_ms=0.0,
            detections=[]
        )

        self.sequence = 0

        # Metrics.
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
            and self.thread.is_alive()
        ):

            self.thread.join(
                timeout=2.0
            )

    def submit_frame(
        self,
        frame: np.ndarray
    ):

        # Only store the newest frame.
        # Old unprocessed frames are deliberately dropped.
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
                    for e in self.whitelist_embeddings
                ]

                threshold = self.threshold

            if frame is None:

                time.sleep(0.002)

                continue

            # ------------------------------------------------
            # Full InsightFace inference happens ONLY here.
            # ------------------------------------------------

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
                - infer_start
            ) * 1000.0

            # ------------------------------------------------
            # Publish result.
            # ------------------------------------------------

            with self.lock:

                self.sequence += 1

                self.latest_result = AIResult(
                    seq_id=self.sequence,
                    timestamp=time.time(),
                    latency_ms=latency_ms,
                    detections=detections
                )

                self.latency_ms = latency_ms

            # ------------------------------------------------
            # AI FPS metric.
            # ------------------------------------------------

            self._count += 1

            now = time.perf_counter()

            elapsed = (
                now - self._fps_timer
            )

            if elapsed >= 1.0:

                self.fps = (
                    self._count
                    /
                    elapsed
                )

                self._count = 0

                self._fps_timer = now

            # ------------------------------------------------
            # Pacing.
            # ------------------------------------------------

            work_time = (
                time.perf_counter()
                - loop_start
            )

            remaining = (
                self.target_interval
                - work_time
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )


# ============================================================
# 7. HYBRID 60 FPS TRACKER
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

        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])

        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

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

        area_a = max(
            0.0,
            a[2] - a[0]
        ) * max(
            0.0,
            a[3] - a[1]
        )

        area_b = max(
            0.0,
            b[2] - b[0]
        ) * max(
            0.0,
            b[3] - b[1]
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return float(
            intersection / union
        )

    def step(
        self,
        ai_result: AIResult,
        now: float
    ):

        # ----------------------------------------------------
        # AI calibration only when a NEW result arrives.
        # ----------------------------------------------------

        if (
            ai_result.seq_id
            >
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

            # No new AI result:
            # move tracks every video frame.
            self._predict_one_frame(
                now
            )

    # --------------------------------------------------------
    # AI association
    # --------------------------------------------------------

    def _apply_ai_result(
        self,
        result: AIResult,
        now: float
    ):

        detections = result.detections

        matched_tracks = set()

        matched_detections = set()

        candidates = []

        # ----------------------------------------------------
        # Generate candidates.
        #
        # A match can happen through:
        #   1. spatial proximity
        #   2. embedding re-identification
        #
        # Embedding is especially useful when a person moves
        # too far between two AI frames.
        # ----------------------------------------------------

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
                        det_center
                        -
                        track_center
                    )
                )

                embedding_similarity = (
                    cosine_similarity(
                        detection.embedding,
                        track.embedding
                    )
                )

                # Spatial match.
                spatial_ok = (
                    distance
                    <=
                    MAX_MATCH_DISTANCE
                )

                # Identity re-identification match.
                reid_ok = (
                    embedding_similarity
                    >=
                    REID_THRESHOLD
                )

                if not spatial_ok and not reid_ok:
                    continue

                # Lower score is better.
                #
                # Strong embedding similarity gets a major
                # advantage, which allows fast-moving faces
                # to reconnect with their old track ID.
                embedding_cost = (
                    1.0
                    -
                    max(
                        -1.0,
                        min(
                            1.0,
                            embedding_similarity
                        )
                    )
                )

                normalized_distance = min(
                    distance
                    /
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

                # If spatially close, prefer it slightly.
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

        # ----------------------------------------------------
        # Consume best matches.
        # ----------------------------------------------------

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
                new_center
                -
                old_center
            )

            # Velocity is expressed in video frames,
            # not AI frames.
            #
            # Because AI may be ~8 FPS while output is 60 FPS,
            # estimate velocity using elapsed time between
            # AI observations.
            elapsed_frames = max(
                1.0,
                track.lost_frames
                +
                1.0
            )

            measured_velocity = (
                movement
                /
                elapsed_frames
            )

            track.velocity = (
                (
                    1.0
                    -
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

            # Smooth embedding update.
            track.embedding = (
                0.70 * track.embedding
                +
                0.30 * detection.embedding
            )

            # Normalize embedding.
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

            # ------------------------------------------------
            # Identity safety.
            #
            # PRIMARY status is only granted when this actual
            # AI detection matches the enrolled embedding.
            # ------------------------------------------------

            if detection.whitelisted:

                track.whitelisted = True

                track.last_verified_time = (
                    now
                )

            else:

                track.whitelisted = False

            matched_tracks.add(
                track_id
            )

            matched_detections.add(
                di
            )

        # ----------------------------------------------------
        # Create tracks for genuinely new detections.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Handle tracks not seen by this AI result.
        # ----------------------------------------------------

        dead = []

        for track_id, track in (
            self.active_tracks.items()
        ):

            if track_id in matched_tracks:
                continue

            track.lost_frames += 1

            track.predict_frames += 1

            # Identity expires if AI has not reverified it.
            if (
                track.whitelisted
                and
                (
                    now
                    -
                    track.last_verified_time
                    >
                    PRIMARY_REVERIFY_TIMEOUT
                )
            ):

                track.whitelisted = False

            # Short prediction period.
            #
            # This helps smooth motion between AI frames.
            # After MAX_PREDICT_FRAMES we stop moving the box
            # so it cannot drift/fly away indefinitely.
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

                # No more extrapolation.
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

    # --------------------------------------------------------
    # Every-video-frame prediction.
    # --------------------------------------------------------

    def _predict_one_frame(
        self,
        now: float
    ):

        dead = []

        for track_id, track in (
            self.active_tracks.items()
        ):

            # Already updated by AI this cycle?
            # This method is only called when there is no new AI
            # result, so every active track gets one frame step.

            track.lost_frames += 1

            track.predict_frames += 1

            if (
                track.whitelisted
                and
                (
                    now
                    -
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
# 8. EMOJI PRIVACY MASK
# ============================================================

# OpenCV's built-in fonts do not reliably render Unicode emoji.
# Pillow is used only once to create a reusable emoji image.
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    raise ImportError(
        "Pillow is required for emoji masking. "
        "Install it with: python -m pip install pillow"
    ) from e

EMOJI_TEXT = "😎"
EMOJI_FONT_PATH = r"C:\Windows\Fonts\seguiemj.ttf"
EMOJI_FONT_SIZE = 256

if not os.path.exists(EMOJI_FONT_PATH):
    raise FileNotFoundError(
        "Segoe UI Emoji font not found at "
        r"C:\Windows\Fonts\seguiemj.ttf"
    )

_emoji_font = ImageFont.truetype(
    EMOJI_FONT_PATH,
    EMOJI_FONT_SIZE
)

_emoji_img = Image.new(
    "RGBA",
    (320, 320),
    (0, 0, 0, 0)
)
_emoji_draw = ImageDraw.Draw(_emoji_img)
_emoji_bbox = _emoji_draw.textbbox(
    (0, 0),
    EMOJI_TEXT,
    font=_emoji_font
)
_emoji_w = _emoji_bbox[2] - _emoji_bbox[0]
_emoji_h = _emoji_bbox[3] - _emoji_bbox[1]
_emoji_draw.text(
    (
        (320 - _emoji_w) // 2 - _emoji_bbox[0],
        (320 - _emoji_h) // 2 - _emoji_bbox[1]
    ),
    EMOJI_TEXT,
    font=_emoji_font,
    embedded_color=True
)
EMOJI_TEMPLATE = np.asarray(_emoji_img, dtype=np.uint8)


def apply_emoji_mask(frame: np.ndarray, bbox: np.ndarray):
    """Cover a face with a centered emoji overlay."""

    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, min(frame_w - 1, int(bbox[0])))
    y1 = max(0, min(frame_h - 1, int(bbox[1])))
    x2 = max(0, min(frame_w, int(bbox[2])))
    y2 = max(0, min(frame_h, int(bbox[3])))

    if x2 <= x1 or y2 <= y1:
        return

    face_w = x2 - x1
    face_h = y2 - y1

    # Slight padding prevents face edges from being visible.
    pad_x = max(2, int(face_w * 0.10))
    pad_y = max(2, int(face_h * 0.10))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)

    target_w = x2 - x1
    target_h = y2 - y1

    scale = min(
        target_w / EMOJI_TEMPLATE.shape[1],
        target_h / EMOJI_TEMPLATE.shape[0]
    )

    emoji_w = max(1, int(EMOJI_TEMPLATE.shape[1] * scale))
    emoji_h = max(1, int(EMOJI_TEMPLATE.shape[0] * scale))

    emoji = cv2.resize(
        EMOJI_TEMPLATE,
        (emoji_w, emoji_h),
        interpolation=cv2.INTER_AREA
    )

    px = x1 + (target_w - emoji_w) // 2
    py = y1 + (target_h - emoji_h) // 2

    px2 = min(frame_w, px + emoji_w)
    py2 = min(frame_h, py + emoji_h)

    if px >= px2 or py >= py2:
        return

    sx2 = px2 - px
    sy2 = py2 - py

    rgb = emoji[:sy2, :sx2, :3]
    alpha = (
        emoji[:sy2, :sx2, 3].astype(np.float32) / 255.0
    )[..., None]

    roi = frame[py:py2, px:px2]

    blended = (
        roi.astype(np.float32) * (1.0 - alpha)
        + rgb.astype(np.float32) * alpha
    )

    frame[py:py2, px:px2] = np.clip(
        blended, 0, 255
    ).astype(np.uint8)


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

        # Embeddings of bystanders manually revealed by the user.
        # These persist even if the tracking ID disappears, so a
        # revealed person stays revealed after leaving and re-entering.
        self.manual_revealed_embeddings: List[np.ndarray] = []

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
        """
        Toggle manual reveal/mask for the face that was clicked.

        First click on a masked bystander:
            emoji -> face revealed

        Second click on the same bystander:
            face revealed -> emoji

        PRIMARY users remain governed by identity verification and
        are not manually masked by this toggle.
        """

        with self.lock:
            coords = self.clicked_coords
            self.clicked_coords = None

        if coords is None:
            return

        cx, cy = coords

        clicked_track = None
        best_distance = float("inf")

        # Find the track whose current bounding box contains the click.
        for track in tracker.active_tracks.values():

            x1, y1, x2, y2 = track.bbox

            if x1 <= cx <= x2 and y1 <= cy <= y2:
                center = tracker.center(track.bbox)
                distance = float(
                    np.linalg.norm(
                        np.array([cx, cy], dtype=np.float32) - center
                    )
                )

                if distance < best_distance:
                    best_distance = distance
                    clicked_track = track

        if clicked_track is None:
            print("[CLICK] No tracked face at clicked position.")
            return

        # PRIMARY identity should stay visible.
        if clicked_track.whitelisted:
            print(
                f"[CLICK] Track {clicked_track.track_id} is PRIMARY; "
                "manual masking is disabled for the primary user."
            )
            return

        # Check whether this person is ALREADY manually revealed.
        # This includes a persistent embedding match from a previous
        # track, which is important after the person leaves and re-enters.
        persistent_reveal_index = None
        persistent_reveal_similarity = -1.0

        for i, saved_embedding in enumerate(
            self.manual_revealed_embeddings
        ):

            similarity = cosine_similarity(
                clicked_track.embedding,
                saved_embedding
            )

            if similarity > persistent_reveal_similarity:
                persistent_reveal_similarity = similarity
                persistent_reveal_index = i

        currently_revealed = (
            clicked_track.manual_revealed
            or
            (
                persistent_reveal_index is not None
                and
                persistent_reveal_similarity >= REID_THRESHOLD
            )
        )

        if currently_revealed:

            # Toggle revealed -> MASKED.
            clicked_track.manual_revealed = False

            # Remove the persistent embedding so this person will be
            # masked again if they leave and re-enter.
            if (
                persistent_reveal_index is not None
                and
                persistent_reveal_similarity >= REID_THRESHOLD
            ):
                self.manual_revealed_embeddings.pop(
                    persistent_reveal_index
                )

            print(
                f"[CLICK] Track {clicked_track.track_id}: "
                "MASKED again. Persistent reveal removed."
            )

        else:

            # Toggle masked -> REVEALED.
            clicked_track.manual_revealed = True

            # Save the embedding so the reveal survives track deletion
            # and re-identification after the person re-enters.
            self.manual_revealed_embeddings.append(
                clicked_track.embedding.copy()
            )

            print(
                f"[CLICK] Track {clicked_track.track_id}: "
                "MANUALLY REVEALED. Reveal persists after re-entry."
            )

    def reset(
        self,
        tracker: HybridTracker,
        ai_worker: AsyncAIWorker
    ):

        self.whitelist_embeddings.clear()
        self.manual_revealed_embeddings.clear()

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
            now
            -
            self._camera_timer
        )

        if elapsed >= 1.0:

            self.camera_fps = (
                self._camera_count
                /
                elapsed
            )

            self._camera_count = 0

            self._camera_timer = now

    def tick_output(self):

        self._output_count += 1

        now = time.perf_counter()

        elapsed = (
            now
            -
            self._output_timer
        )

        if elapsed >= 1.0:

            self.output_fps = (
                self._output_count
                /
                elapsed
            )

            self._output_count = 0

            self._output_timer = now

    def tick_loop(self):

        self._loop_count += 1

        now = time.perf_counter()

        elapsed = (
            now
            -
            self._loop_timer
        )

        if elapsed >= 1.0:

            self.main_loop_fps = (
                self._loop_count
                /
                elapsed
            )

            self._loop_count = 0

            self._loop_timer = now


# ============================================================
# 11. MAIN
# ============================================================


# ============================================================
# 11. DISK-BACKED TEMPORAL VIDEO BUFFER
# ============================================================

BUFFER_DELAY_SECONDS = 30.0
SEGMENT_SECONDS = 1.0
BUFFER_CODEC = "MJPG"


@dataclass
class VideoSegment:
    sequence: int
    path: str
    end_time: float


class DiskVideoBuffer:
    """Stores short encoded video segments on disk instead of RAM."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.segments: List[VideoSegment] = []
        self.next_to_process: Optional[int] = None

        self.latest_live_frame: Optional[np.ndarray] = None
        self.running = False
        self.capture_error: Optional[str] = None

        self.width = 0
        self.height = 0
        self.fps = TARGET_FPS

        self.thread = None

    def start(self, camera_index=0):
        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_index,),
            daemon=True,
            name="PrivaStream-CaptureBuffer"
        )
        self.thread.start()

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

    def get_live_frame(self):
        with self.lock:
            return (
                None
                if self.latest_live_frame is None
                else self.latest_live_frame.copy()
            )

    def get_status(self):
        with self.lock:
            return len(self.segments), self.capture_error

    def get_ready_segment(self, delay_seconds):
        now = time.time()

        with self.condition:
            if self.next_to_process is None:
                for segment in self.segments:
                    if segment.end_time <= now - delay_seconds:
                        self.next_to_process = segment.sequence
                        break

            if self.next_to_process is None:
                return None

            for segment in self.segments:
                if segment.sequence == self.next_to_process:
                    return segment

        return None

    def mark_processed(self, sequence):
        path = None

        with self.condition:
            for i, segment in enumerate(self.segments):
                if segment.sequence == sequence:
                    path = segment.path
                    self.segments.pop(i)
                    break

            if self.next_to_process == sequence:
                self.next_to_process = None

            self.condition.notify_all()

        if path:
            try:
                os.remove(path)
            except OSError:
                pass

    def _capture_loop(self, camera_index):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            self.capture_error = "Could not open camera."
            self.running = False
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS

        frames_per_segment = max(
            1,
            int(round(self.fps * SEGMENT_SECONDS))
        )

        fourcc = cv2.VideoWriter_fourcc(*BUFFER_CODEC)

        writer = None
        current_path = None
        frame_count = 0
        sequence = 0

        print(
            f"[BUFFER] Camera: {self.width}x{self.height} "
            f"@ {self.fps:.1f} FPS"
        )
        print(
            f"[BUFFER] Delay: {BUFFER_DELAY_SECONDS:.0f}s | "
            f"Segment: {SEGMENT_SECONDS:.0f}s | Codec: {BUFFER_CODEC}"
        )
        print(f"[BUFFER] Disk: {self.directory}")

        try:
            while self.running:
                ret, frame = cap.read()

                if not ret:
                    self.capture_error = "Camera frame capture failed."
                    break

                with self.lock:
                    self.latest_live_frame = frame.copy()

                if writer is None:
                    current_path = os.path.join(
                        self.directory,
                        f"segment_{sequence:08d}.avi"
                    )

                    writer = cv2.VideoWriter(
                        current_path,
                        fourcc,
                        self.fps,
                        (self.width, self.height)
                    )

                    if not writer.isOpened():
                        self.capture_error = (
                            "Could not open MJPG buffer writer."
                        )
                        break

                    frame_count = 0

                writer.write(frame)
                frame_count += 1

                if frame_count >= frames_per_segment:
                    writer.release()
                    writer = None

                    segment = VideoSegment(
                        sequence=sequence,
                        path=current_path,
                        end_time=time.time()
                    )

                    with self.condition:
                        self.segments.append(segment)
                        self.condition.notify_all()

                    sequence += 1

        finally:
            if writer is not None:
                writer.release()

            cap.release()

            with self.condition:
                self.running = False
                self.condition.notify_all()


# ============================================================
# 12. DELAYED PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("PrivaStream - SAMPLE 7")
    print("30-Second Disk Buffer + Privacy Processing + OBS")
    print("=" * 70)

    app, active_provider = initialize_insightface()

    ai_worker = AsyncAIWorker(app, AI_TARGET_FPS)
    tracker = HybridTracker()
    enrollment = EnrollmentManager()
    metrics = PerformanceMetrics()

    buffer_dir = os.path.join(
        os.environ.get("TEMP", "."),
        "PrivaStreamBuffer"
    )

    video_buffer = DiskVideoBuffer(buffer_dir)
    video_buffer.start(camera_index=0)

    while video_buffer.width == 0 and video_buffer.running:
        time.sleep(0.05)

    actual_width = video_buffer.width or CAMERA_WIDTH
    actual_height = video_buffer.height or CAMERA_HEIGHT
    actual_camera_fps = video_buffer.fps or TARGET_FPS

    live_window = "PrivaStream - LIVE CAMERA"
    delayed_window = "PrivaStream - DELAYED OUTPUT"

    cv2.namedWindow(live_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(delayed_window, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(
        live_window,
        actual_width // 2,
        actual_height // 2
    )
    cv2.resizeWindow(
        delayed_window,
        actual_width,
        actual_height
    )

    cv2.setMouseCallback(
        delayed_window,
        enrollment.on_mouse_click
    )

    vcam = None

    try:
        vcam = pyvirtualcam.Camera(
            width=actual_width,
            height=actual_height,
            fps=TARGET_FPS,
            fmt=pyvirtualcam.PixelFormat.BGR
        )

        print(f"[VCAM] Virtual camera: {vcam.device}")

    except Exception as e:
        video_buffer.stop()
        cv2.destroyAllWindows()
        raise RuntimeError(
            f"Could not start virtual camera: {e}"
        )

    ai_worker.start()

    print()
    print("[CONTROLS]")
    print("  Click delayed face -> reveal / mask toggle")
    print("  R                  -> reset identities")
    print("  Q                  -> quit")
    print()
    print(
        f"[BUFFER] Filling {BUFFER_DELAY_SECONDS:.0f}-second buffer..."
    )

    processed_frames = 0

    try:

        while video_buffer.running:

            # ------------------------------------------------
            # LIVE CAMERA WINDOW
            # ------------------------------------------------

            live_frame = video_buffer.get_live_frame()

            if live_frame is not None:
                live_display = live_frame.copy()

                cv2.putText(
                    live_display,
                    "LIVE CAMERA",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    live_window,
                    live_display
                )

            # ------------------------------------------------
            # GET FRAME SEGMENT AFTER DELAY
            # ------------------------------------------------

            segment = video_buffer.get_ready_segment(
                BUFFER_DELAY_SECONDS
            )

            if segment is None:

                count, error = video_buffer.get_status()

                if error:
                    raise RuntimeError(error)

                cv2.putText(
                    live_display if live_frame is not None else np.zeros(
                        (actual_height, actual_width, 3),
                        dtype=np.uint8
                    ),
                    f"Buffering... {count}s / "
                    f"{BUFFER_DELAY_SECONDS:.0f}s",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                continue

            # ------------------------------------------------
            # PROCESS EVERY FRAME IN THE DELAYED SEGMENT
            # ------------------------------------------------

            segment_cap = cv2.VideoCapture(segment.path)

            if not segment_cap.isOpened():
                print(
                    f"[BUFFER] Failed to open {segment.path}"
                )
                video_buffer.mark_processed(segment.sequence)
                continue

            while True:

                ret, frame = segment_cap.read()

                if not ret:
                    break

                frame_start = time.perf_counter()

                # AI worker receives every frame, but only runs
                # inference at AI_TARGET_FPS and keeps the newest frame.
                ai_worker.submit_frame(frame)

                ai_result = ai_worker.get_latest_result()

                now = time.time()

                ai_age = (
                    now - ai_result.timestamp
                    if ai_result.timestamp > 0
                    else float("inf")
                )

                tracker_start = time.perf_counter()

                tracker.step(
                    ai_result,
                    now
                )

                tracking_ms = (
                    time.perf_counter()
                    - tracker_start
                ) * 1000.0

                enrollment.process_click(
                    tracker,
                    ai_result,
                    ai_worker
                )

                failsafe_active = (
                    ai_result.seq_id == 0
                    or ai_age > FAILSAFE_TIMEOUT
                )

                output_frame = frame.copy()

                emoji_start = time.perf_counter()

                for track in list(
                    tracker.active_tracks.values()
                ):

                    persistent_manual_reveal = False

                    if not track.whitelisted:

                        for saved_embedding in (
                            enrollment.manual_revealed_embeddings
                        ):

                            if cosine_similarity(
                                track.embedding,
                                saved_embedding
                            ) >= REID_THRESHOLD:

                                persistent_manual_reveal = True
                                break

                    if (
                        failsafe_active
                        or (
                            not track.whitelisted
                            and not track.manual_revealed
                            and not persistent_manual_reveal
                        )
                    ):

                        apply_emoji_mask(
                            output_frame,
                            track.bbox
                        )

                emoji_ms = (
                    time.perf_counter()
                    - emoji_start
                ) * 1000.0

                # ------------------------------------------------
                # OBS OUTPUT
                # ------------------------------------------------

                vcam.send(output_frame)
                metrics.tick_output()
                processed_frames += 1

                # ------------------------------------------------
                # DELAYED OUTPUT WINDOW
                # ------------------------------------------------

                display_frame = output_frame.copy()

                hud_lines = [
                    f"OUTPUT: {metrics.output_fps:.1f} FPS",
                    f"CAMERA: {actual_camera_fps:.1f} FPS",
                    f"AI: {ai_worker.fps:.1f} FPS ({ai_worker.latency_ms:.1f} ms)",
                    f"TRACK: {tracking_ms:.2f} ms",
                    f"EMOJI: {emoji_ms:.2f} ms",
                    f"FACES: {len(tracker.active_tracks)}",
                    f"ENGINE: {'CUDA' if active_provider == 'CUDAExecutionProvider' else 'CPU'}",
                    f"DELAY: {BUFFER_DELAY_SECONDS:.0f} SEC",
                    "STATE: FAIL-SAFE" if failsafe_active else "STATE: PRIVACY ACTIVE",
                ]

                hud_height = 18 + len(hud_lines) * 23

                cv2.rectangle(
                    display_frame,
                    (10, 10),
                    (390, hud_height),
                    (0, 0, 0),
                    -1
                )

                y = 31

                for line in hud_lines:

                    if line.startswith("STATE:"):
                        hud_color = (
                            (0, 165, 255)
                            if failsafe_active
                            else (0, 255, 0)
                        )
                        thickness = 2

                    elif line.startswith("OUTPUT:"):
                        hud_color = (255, 255, 0)
                        thickness = 2

                    else:
                        hud_color = (220, 220, 220)
                        thickness = 1

                    cv2.putText(
                        display_frame,
                        line,
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        hud_color,
                        thickness
                    )

                    y += 23

                cv2.imshow(
                    delayed_window,
                    display_frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    video_buffer.running = False
                    break

                elif key == ord("r"):
                    enrollment.reset(
                        tracker,
                        ai_worker
                    )

                # Keep the processed stream paced at 60 FPS.
                elapsed = time.perf_counter() - frame_start
                target_time = 1.0 / TARGET_FPS

                if elapsed < target_time:
                    time.sleep(
                        target_time - elapsed
                    )

            segment_cap.release()

            video_buffer.mark_processed(
                segment.sequence
            )

    except KeyboardInterrupt:
        print("\n[SYSTEM] Keyboard interrupt.")

    finally:

        print("[SYSTEM] Cleaning up...")

        ai_worker.stop()

        if vcam is not None:
            try:
                vcam.close()
            except Exception:
                pass

        video_buffer.stop()

        cv2.destroyAllWindows()

        print(
            f"[SYSTEM] Processed {processed_frames} frames."
        )
        print("[SYSTEM] PrivaStream stopped cleanly.")


if __name__ == "__main__":
    main()
