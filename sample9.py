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
# 11. STREAMER-FRIENDLY DELAYED PIPELINE
# ============================================================

# The delay is a processing window, NOT a UI buffering loop.
# LIVE CAMERA stays visible continuously.
# Once the delay is filled, the processed output also stays live
# and never goes blank just because the next segment is loading.
BUFFER_DELAY_SECONDS = 30.0

# Analyze a face every N frames. Tracking fills the intermediate
# frames. 5 at 60 FPS ~= 12 AI opportunities/sec.
AI_SAMPLE_EVERY = 5

SEGMENT_SECONDS = 1.0
BUFFER_CODEC = "MJPG"


@dataclass
class BufferedSegment:
    sequence: int
    path: str
    start_time: float
    end_time: float


class TemporalCameraBuffer:
    """
    Captures the physical camera continuously.

    The camera preview is ALWAYS available.
    Encoded one-second segments are written to disk so a 30-sec
    delay does not consume several GB of RAM.
    """

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
            target=self._capture_loop,
            args=(camera_index,),
            daemon=True,
            name="PrivaStream-Camera"
        )
        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

    def get_live_frame(self):
        with self.lock:
            if self.latest_live_frame is None:
                return None
            return self.latest_live_frame.copy()

    def get_ready_segments(self):
        cutoff = time.time() - BUFFER_DELAY_SECONDS

        with self.lock:
            ready = [
                s for s in self.segments
                if s.end_time <= cutoff
            ]

        return sorted(
            ready,
            key=lambda s: s.sequence
        )

    def remove_segment(self, sequence):
        path = None

        with self.lock:
            remaining = []

            for segment in self.segments:
                if segment.sequence == sequence:
                    path = segment.path
                else:
                    remaining.append(segment)

            self.segments = remaining

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
                -
                self.segments[0].start_time
            )

    def _capture_loop(self, camera_index):

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

        self.fps = (
            cap.get(cv2.CAP_PROP_FPS)
            or TARGET_FPS
        )

        frames_per_segment = max(
            1,
            int(round(
                self.fps * SEGMENT_SECONDS
            ))
        )

        fourcc = cv2.VideoWriter_fourcc(
            *BUFFER_CODEC
        )

        sequence = 0
        writer = None
        frame_count = 0
        segment_start = 0.0
        current_path = None

        print(
            f"[CAMERA] {self.width}x{self.height} "
            f"@ {self.fps:.1f} FPS"
        )

        print(
            f"[BUFFER] {BUFFER_DELAY_SECONDS:.0f}s "
            f"disk-backed temporal buffer"
        )

        try:

            while self.running:

                ret, frame = cap.read()

                if not ret:
                    self.error = (
                        "Camera frame capture failed."
                    )
                    break

                now = time.time()

                # This is what makes the LIVE window genuinely live.
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

                        self.error = (
                            "Could not create disk buffer."
                        )
                        break

                    frame_count = 0
                    segment_start = now

                writer.write(frame)
                frame_count += 1

                if frame_count >= frames_per_segment:

                    writer.release()
                    writer = None

                    with self.lock:
                        self.segments.append(
                            BufferedSegment(
                                sequence=sequence,
                                path=current_path,
                                start_time=segment_start,
                                end_time=now
                            )
                        )

                    sequence += 1

        finally:

            if writer is not None:
                writer.release()

            cap.release()
            self.running = False


class LiveFaceSelector:

    """
    Mouse selection happens on the LIVE camera.

    The click is matched against the newest AI detections.
    The selected embedding is then enrolled as the primary
    identity, so the delayed output can use it.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.pending_click = None

    def callback(
        self,
        event,
        x,
        y,
        flags,
        param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            with self.lock:
                self.pending_click = (x, y)

    def consume_click(self):

        with self.lock:
            click = self.pending_click
            self.pending_click = None
            return click


def enroll_from_live_click(
    selector,
    ai_result,
    enrollment,
    ai_worker
):

    click = selector.consume_click()

    if click is None:
        return

    cx, cy = click

    # Find the freshest AI face covering the live click.
    best = None
    best_distance = float("inf")

    for detection in ai_result.detections:

        x1, y1, x2, y2 = detection.bbox

        if (
            x1 <= cx <= x2
            and
            y1 <= cy <= y2
        ):

            center = np.array(
                [
                    (x1 + x2) * 0.5,
                    (y1 + y2) * 0.5
                ],
                dtype=np.float32
            )

            distance = float(
                np.linalg.norm(
                    center
                    -
                    np.array(
                        [cx, cy],
                        dtype=np.float32
                    )
                )
            )

            if distance < best_distance:
                best = detection
                best_distance = distance

    if best is None:

        print(
            "[LIVE SELECT] No AI face matched the "
            "live click. Try clicking the face again."
        )
        return

    # Add/update primary identity.
    enrollment.whitelist_embeddings = [
        best.embedding.copy()
    ]

    ai_worker.update_whitelist(
        enrollment.whitelist_embeddings
    )

    print(
        "[LIVE SELECT] Face selected as PRIMARY. "
        f"Similarity: {best.similarity:.3f}"
    )


class StreamProcessor:

    """
    Converts delayed buffered video into a continuous processed
    stream.

    Critical behavior:
      - It never sends raw frames to OBS.
      - It never intentionally blanks the output after startup.
      - It keeps the last processed frame until a new processed
        frame is ready.
      - AI is sampled periodically while tracking runs for every
        frame.
    """

    def __init__(
        self,
        camera_buffer,
        ai_worker,
        tracker,
        enrollment
    ):

        self.camera_buffer = camera_buffer
        self.ai_worker = ai_worker
        self.tracker = tracker
        self.enrollment = enrollment

        self.running = False
        self.thread = None

        self.output_lock = threading.Lock()

        self.output_queue = []
        self.last_output = None

        self.started_output = False
        self.processing_fps = 0.0
        self.dropped_segments = 0

    def start(self):
        self.running = True

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="PrivaStream-Processor"
        )

        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def get_output(self):

        with self.output_lock:

            if self.output_queue:

                frame = self.output_queue.pop(0)

                return frame

            if self.last_output is not None:

                # Returning the last processed frame prevents
                # a blank screen once the pipeline is running.
                return self.last_output.copy()

        return None

    def _render_privacy(
        self,
        frame,
        ai_result
    ):

        now = time.time()

        ai_age = (
            now - ai_result.timestamp
            if ai_result.timestamp > 0
            else float("inf")
        )

        # During normal operation the processor is working on
        # delayed footage, so we don't use a stale wall-clock
        # check to randomly blank the output.
        #
        # Before the first AI result, conservative masking applies.
        no_identity_yet = (
            ai_result.seq_id == 0
        )

        output = frame.copy()

        for track in list(
            self.tracker.active_tracks.values()
        ):

            persistent_manual_reveal = False

            if not track.whitelisted:

                for saved_embedding in (
                    self.enrollment
                    .manual_revealed_embeddings
                ):

                    if cosine_similarity(
                        track.embedding,
                        saved_embedding
                    ) >= REID_THRESHOLD:

                        persistent_manual_reveal = True
                        break

            should_mask = (
                no_identity_yet
                or
                (
                    not track.whitelisted
                    and
                    not track.manual_revealed
                    and
                    not persistent_manual_reveal
                )
            )

            if should_mask:

                apply_emoji_mask(
                    output,
                    track.bbox
                )

        return output

    def _loop(self):

        processed_sequences = set()
        frame_number = 0

        fps_count = 0
        fps_timer = time.perf_counter()

        while self.running:

            segments = (
                self.camera_buffer
                .get_ready_segments()
            )

            segment = None

            for candidate in segments:

                if (
                    candidate.sequence
                    not in processed_sequences
                ):

                    segment = candidate
                    break

            if segment is None:

                time.sleep(0.002)
                continue

            cap = cv2.VideoCapture(
                segment.path
            )

            if not cap.isOpened():

                self.camera_buffer.remove_segment(
                    segment.sequence
                )

                processed_sequences.add(
                    segment.sequence
                )

                continue

            while self.running:

                ret, frame = cap.read()

                if not ret:
                    break

                # Expensive AI is sampled at a higher rate than
                # the old 3 FPS design. Tracking fills every
                # intermediate frame.
                if (
                    frame_number
                    %
                    AI_SAMPLE_EVERY
                    == 0
                ):

                    self.ai_worker.submit_frame(
                        frame
                    )

                ai_result = (
                    self.ai_worker.get_latest_result()
                )

                now = time.time()

                self.tracker.step(
                    ai_result,
                    now
                )

                # Delayed-output clicks are intentionally NOT used
                # for face selection. Selection is done on LIVE.
                output = self._render_privacy(
                    frame,
                    ai_result
                )

                with self.output_lock:

                    self.output_queue.append(
                        output
                    )

                    self.last_output = (
                        output.copy()
                    )

                    # A small queue is enough because the main
                    # output loop drains it at 60 FPS.
                    if len(
                        self.output_queue
                    ) > 180:

                        del self.output_queue[
                            :len(
                                self.output_queue
                            ) - 180
                        ]

                    self.started_output = True

                frame_number += 1
                fps_count += 1

                elapsed = (
                    time.perf_counter()
                    -
                    fps_timer
                )

                if elapsed >= 1.0:

                    self.processing_fps = (
                        fps_count
                        /
                        elapsed
                    )

                    fps_count = 0
                    fps_timer = (
                        time.perf_counter()
                    )

            cap.release()

            processed_sequences.add(
                segment.sequence
            )

            self.camera_buffer.remove_segment(
                segment.sequence
            )


# ============================================================
# 14. SAMPLE 9 MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PrivaStream - SAMPLE 9")
    print("LIVE SELECT + 30s Delayed Privacy Output")
    print("=" * 70)

    app, active_provider = (
        initialize_insightface()
    )

    ai_worker = AsyncAIWorker(
        app,
        8.0
    )

    tracker = HybridTracker()
    enrollment = EnrollmentManager()
    selector = LiveFaceSelector()

    camera_buffer = TemporalCameraBuffer(
        os.path.join(
            os.environ.get(
                "TEMP",
                "."
            ),
            "PrivaStreamBuffer"
        )
    )

    camera_buffer.start(
        camera_index=0
    )

    # Wait ONLY for camera initialization.
    # This is not the 30-sec UI buffer.
    while (
        camera_buffer.width == 0
        and
        camera_buffer.running
    ):

        time.sleep(0.05)

    if camera_buffer.error:

        camera_buffer.stop()

        raise RuntimeError(
            camera_buffer.error
        )

    width = camera_buffer.width
    height = camera_buffer.height
    fps = camera_buffer.fps

    live_window = (
        "PrivaStream - LIVE CAMERA"
    )

    output_window = (
        "PrivaStream - PROCESSED OUTPUT"
    )

    cv2.namedWindow(
        live_window,
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        output_window,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        live_window,
        width // 2,
        height // 2
    )

    cv2.resizeWindow(
        output_window,
        width,
        height
    )

    # IMPORTANT:
    # Clicking is done on the LIVE camera.
    cv2.setMouseCallback(
        live_window,
        selector.callback
    )

    vcam = None

    try:

        vcam = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=TARGET_FPS,
            fmt=pyvirtualcam.PixelFormat.BGR
        )

        print(
            f"[VCAM] Virtual camera: {vcam.device}"
        )

    except Exception as e:

        camera_buffer.stop()
        cv2.destroyAllWindows()

        raise RuntimeError(
            f"Could not start virtual camera: {e}"
        )

    ai_worker.start()

    processor = StreamProcessor(
        camera_buffer,
        ai_worker,
        tracker,
        enrollment
    )

    processor.start()

    print()
    print("[CONTROLS]")
    print(
        "  LIVE window: click a face -> select PRIMARY"
    )
    print(
        "  R -> reset primary identity"
    )
    print(
        "  Q -> quit"
    )
    print()
    print(
        "[SYSTEM] LIVE camera is available immediately."
    )
    print(
        "[SYSTEM] Delayed output will appear once "
        f"{BUFFER_DELAY_SECONDS:.0f}s of footage exists."
    )

    output_started = False
    output_clock = time.perf_counter()
    output_interval = 1.0 / TARGET_FPS
    next_output = output_clock

    try:

        while camera_buffer.running:

            # =================================================
            # 1. LIVE CAMERA — ALWAYS DISPLAYED
            # =================================================

            live_frame = (
                camera_buffer.get_live_frame()
            )

            if live_frame is not None:

                live_display = (
                    live_frame.copy()
                )

                cv2.rectangle(
                    live_display,
                    (10, 10),
                    (470, 65),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    live_display,
                    "LIVE CAMERA  |  CLICK FACE TO SELECT",
                    (20, 47),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    live_window,
                    live_display
                )

                # A live click is converted into a primary
                # identity using the latest AI observation.
                latest_ai = (
                    ai_worker.get_latest_result()
                )

                enroll_from_live_click(
                    selector,
                    latest_ai,
                    enrollment,
                    ai_worker
                )

            # =================================================
            # 2. PROCESSED DELAYED OUTPUT
            # =================================================

            output = processor.get_output()

            if output is not None:

                output_started = True

                display = output.copy()

                # Minimal, clean HUD.
                cv2.rectangle(
                    display,
                    (10, 10),
                    (360, 115),
                    (0, 0, 0),
                    -1
                )

                lines = [
                    "PRIVASTREAM  |  PRIVACY ACTIVE",
                    f"OUTPUT: {TARGET_FPS:.0f} FPS",
                    f"DELAY: {BUFFER_DELAY_SECONDS:.0f} SEC",
                    "AI: AHEAD-OF-TIME"
                ]

                y = 34

                for line in lines:

                    cv2.putText(
                        display,
                        line,
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (220, 220, 220),
                        1
                    )

                    y += 23

                cv2.imshow(
                    output_window,
                    display
                )

                # ONLY processed output goes to OBS.
                vcam.send(output)

            elif not output_started:

                # Only during the initial 30-sec startup period.
                # After output starts, we never replace the stream
                # with a blank "buffering" screen.
                wait = np.zeros(
                    (height, width, 3),
                    dtype=np.uint8
                )

                elapsed_buffer = (
                    camera_buffer.buffered_seconds()
                )

                progress = min(
                    elapsed_buffer
                    /
                    BUFFER_DELAY_SECONDS,
                    1.0
                )

                cv2.putText(
                    wait,
                    "PrivaStream is preparing the "
                    "privacy-delayed output...",
                    (40, height // 2 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (220, 220, 220),
                    2
                )

                cv2.putText(
                    wait,
                    (
                        f"Privacy delay: "
                        f"{elapsed_buffer:.0f}/"
                        f"{BUFFER_DELAY_SECONDS:.0f} sec"
                    ),
                    (40, height // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv2.imshow(
                    output_window,
                    wait
                )

                # Do NOT send raw/live frames to OBS.
                # Send a safe placeholder only until the
                # privacy pipeline has its first processed frame.
                vcam.send(wait)

            # =================================================
            # 3. CONTROLS
            # =================================================

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == ord("q"):
                break

            elif key == ord("r"):

                enrollment.reset(
                    tracker,
                    ai_worker
                )

            # =================================================
            # 4. OUTPUT CLOCK
            # =================================================

            next_output += output_interval

            sleep_time = (
                next_output
                -
                time.perf_counter()
            )

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_output = (
                    time.perf_counter()
                )

    except KeyboardInterrupt:

        print(
            "\n[SYSTEM] Keyboard interrupt."
        )

    finally:

        print(
            "[SYSTEM] Cleaning up..."
        )

        processor.stop()
        ai_worker.stop()
        camera_buffer.stop()

        if vcam is not None:

            try:
                vcam.close()
            except Exception:
                pass

        cv2.destroyAllWindows()

        print(
            "[SYSTEM] PrivaStream stopped cleanly."
        )


if __name__ == "__main__":
    main()
