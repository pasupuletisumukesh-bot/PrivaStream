import cv2
import numpy as np
import pyvirtualcam
import time
import onnxruntime as ort
from typing import List, Dict, Tuple


# ============================================================
# CUDA / ONNX RUNTIME INITIALIZATION
# ============================================================

print("[INIT] Preloading CUDA/cuDNN DLLs...")

try:
    ort.preload_dlls(directory="")
    print("[INIT] CUDA/cuDNN DLL preload completed.")
except Exception as e:
    print(f"[INIT] CUDA DLL preload warning: {e}")


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

# Using buffalo_s for this experiment
MODEL_NAME = "buffalo_s"

# Face recognition threshold
SIMILARITY_THRESHOLD = 0.44

# Maximum distance for associating detections
MAX_MATCH_DISTANCE = 160.0

# Keep a track alive briefly if detection temporarily fails
MAX_LOST_FRAMES = 8

# Blur optimization
BLUR_DOWNSAMPLE = 4


# ============================================================
# DATA STRUCTURES
# ============================================================

class FaceTrack:

    def __init__(
        self,
        track_id: int,
        bbox: np.ndarray,
        embedding: np.ndarray,
        whitelisted: bool,
        similarity: float
    ):

        self.track_id = track_id

        self.bbox = bbox.copy()

        self.embedding = embedding.copy()

        self.whitelisted = whitelisted

        self.similarity = similarity

        self.lost_frames = 0

        self.velocity = np.zeros(
            2,
            dtype=np.float32
        )


# ============================================================
# GLOBAL STATE
# ============================================================

whitelist_embeddings: List[np.ndarray] = []

active_tracks: Dict[int, FaceTrack] = {}

next_track_id = 0

clicked_coords = None


# ============================================================
# MOUSE CALLBACK
# ============================================================

def on_mouse_click(
    event,
    x,
    y,
    flags,
    param
):

    global clicked_coords

    if event == cv2.EVENT_LBUTTONDOWN:

        clicked_coords = (
            x,
            y
        )


# ============================================================
# FACE SIMILARITY
# ============================================================

def compute_cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:

    denominator = (
        np.linalg.norm(vec1)
        *
        np.linalg.norm(vec2)
        +
        1e-6
    )

    return float(
        np.dot(vec1, vec2)
        /
        denominator
    )


def find_best_whitelist_match(
    embedding: np.ndarray
) -> Tuple[bool, float]:

    if not whitelist_embeddings:

        return False, 0.0

    best_similarity = 0.0

    for saved_embedding in whitelist_embeddings:

        similarity = compute_cosine_similarity(
            embedding,
            saved_embedding
        )

        if similarity > best_similarity:

            best_similarity = similarity

    return (
        best_similarity >= SIMILARITY_THRESHOLD,
        best_similarity
    )


# ============================================================
# GEOMETRY
# ============================================================

def get_centroid(
    bbox: np.ndarray
) -> np.ndarray:

    return np.array(
        [
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0
        ],
        dtype=np.float32
    )


# ============================================================
# FAST PRIVACY BLUR
# ============================================================

def apply_fast_blur(
    frame: np.ndarray,
    bbox: np.ndarray
):

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = [
        int(v)
        for v in bbox
    ]

    # --------------------------------------------------------
    # Clip bounding box
    # --------------------------------------------------------

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
        min(width, x2)
    )

    y2 = max(
        0,
        min(height, y2)
    )

    if x2 <= x1 or y2 <= y1:

        return

    roi = frame[
        y1:y2,
        x1:x2
    ]

    roi_h, roi_w = roi.shape[:2]

    if roi_h < 6 or roi_w < 6:

        return

    # --------------------------------------------------------
    # Downsample
    # --------------------------------------------------------

    small_w = max(
        2,
        roi_w // BLUR_DOWNSAMPLE
    )

    small_h = max(
        2,
        roi_h // BLUR_DOWNSAMPLE
    )

    small = cv2.resize(
        roi,
        (
            small_w,
            small_h
        ),
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

    blurred = cv2.resize(
        blurred_small,
        (
            roi_w,
            roi_h
        ),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # Elliptical feather mask
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Feather mask edges
    # --------------------------------------------------------

    feather_size = min(
        15,
        max(
            5,
            (min(roi_w, roi_h) // 8) | 1
        )
    )

    mask = cv2.GaussianBlur(
        mask,
        (
            feather_size,
            feather_size
        ),
        0
    )

    alpha = (
        mask.astype(np.float32)
        /
        255.0
    )

    alpha = alpha[..., None]

    # --------------------------------------------------------
    # Vectorized blend
    # --------------------------------------------------------

    blended = (
        roi.astype(np.float32)
        *
        (1.0 - alpha)
        +
        blurred.astype(np.float32)
        *
        alpha
    )

    roi[:] = blended.astype(
        np.uint8
    )


# ============================================================
# TRACK ASSOCIATION
# ============================================================

def associate_detections(
    detections: List[dict]
):

    global active_tracks
    global next_track_id

    matched_tracks = set()

    matched_detections = set()

    # --------------------------------------------------------
    # Build possible spatial matches
    # --------------------------------------------------------

    candidate_matches = []

    for det_index, detection in enumerate(
        detections
    ):

        det_center = get_centroid(
            detection["bbox"]
        )

        for track_id, track in active_tracks.items():

            if track_id in matched_tracks:

                continue

            track_center = get_centroid(
                track.bbox
            )

            distance = np.linalg.norm(
                det_center - track_center
            )

            if distance <= MAX_MATCH_DISTANCE:

                candidate_matches.append(
                    (
                        distance,
                        det_index,
                        track_id
                    )
                )

    # --------------------------------------------------------
    # Closest match first
    # --------------------------------------------------------

    candidate_matches.sort(
        key=lambda x: x[0]
    )

    # --------------------------------------------------------
    # Update existing tracks
    # --------------------------------------------------------

    for (
        distance,
        det_index,
        track_id
    ) in candidate_matches:

        if det_index in matched_detections:

            continue

        if track_id in matched_tracks:

            continue

        detection = detections[
            det_index
        ]

        track = active_tracks[
            track_id
        ]

        old_center = get_centroid(
            track.bbox
        )

        new_center = get_centroid(
            detection["bbox"]
        )

        velocity = (
            new_center
            -
            old_center
        )

        track.velocity = (
            track.velocity * 0.65
            +
            velocity * 0.35
        )

        track.bbox = detection[
            "bbox"
        ].copy()

        track.embedding = detection[
            "embedding"
        ].copy()

        track.whitelisted = detection[
            "whitelisted"
        ]

        track.similarity = detection[
            "similarity"
        ]

        track.lost_frames = 0

        matched_tracks.add(
            track_id
        )

        matched_detections.add(
            det_index
        )

    # --------------------------------------------------------
    # Create new tracks
    # --------------------------------------------------------

    for det_index, detection in enumerate(
        detections
    ):

        if det_index in matched_detections:

            continue

        new_track = FaceTrack(
            track_id=next_track_id,
            bbox=detection["bbox"],
            embedding=detection["embedding"],
            whitelisted=detection["whitelisted"],
            similarity=detection["similarity"]
        )

        active_tracks[
            next_track_id
        ] = new_track

        next_track_id += 1

    # --------------------------------------------------------
    # Handle missed detections
    # --------------------------------------------------------

    for track_id in list(
        active_tracks.keys()
    ):

        if track_id not in matched_tracks:

            track = active_tracks[
                track_id
            ]

            track.lost_frames += 1

            if (
                track.lost_frames
                <= MAX_LOST_FRAMES
            ):

                # Very short motion prediction
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

            else:

                # Remove completely lost track
                del active_tracks[
                    track_id
                ]


# ============================================================
# ENROLLMENT
# ============================================================

def process_enrollment():

    global clicked_coords

    if clicked_coords is None:

        return

    x, y = clicked_coords

    clicked_coords = None

    selected_track = None

    # --------------------------------------------------------
    # Find selected face
    # --------------------------------------------------------

    for track in active_tracks.values():

        x1, y1, x2, y2 = track.bbox

        if (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        ):

            selected_track = track

            break

    if selected_track is None:

        print(
            "[ENROLL] No face selected."
        )

        return

    # --------------------------------------------------------
    # Toggle primary status
    # --------------------------------------------------------

    if selected_track.whitelisted:

        selected_track.whitelisted = False

        print(
            f"[ENROLL] Track "
            f"{selected_track.track_id} "
            f"removed from primary."
        )

    else:

        whitelist_embeddings.append(
            selected_track.embedding.copy()
        )

        selected_track.whitelisted = True

        print(
            f"[ENROLL] Track "
            f"{selected_track.track_id} "
            f"registered as primary."
        )


# ============================================================
# RESET
# ============================================================

def reset_permissions():

    global whitelist_embeddings
    global active_tracks
    global next_track_id

    whitelist_embeddings.clear()

    active_tracks.clear()

    next_track_id = 0

    print(
        "[ENROLL] All permissions reset."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)

    print(
        "PrivaStream - buffalo_s Full Per-Frame GPU/CPU Pipeline"
    )

    print("=" * 65)

    # ========================================================
    # INSIGHTFACE
    # ========================================================

    print(
        "[INIT] Loading buffalo_s with GPU/CPU provider detection..."
    )

    available_providers = (
        ort.get_available_providers()
    )

    print(
        "[INIT] ONNX Runtime providers:",
        available_providers
    )

    # --------------------------------------------------------
    # Select CUDA if available
    # --------------------------------------------------------

    if "CUDAExecutionProvider" in available_providers:

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

    # --------------------------------------------------------
    # Create InsightFace application
    # --------------------------------------------------------

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=providers
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    print(
        "[INIT] buffalo_s ready."
    )

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera."
        )

    # Reduce camera buffering
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    # Request MJPG
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    # Resolution
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    # FPS
    cap.set(
        cv2.CAP_PROP_FPS,
        TARGET_FPS
    )

    # --------------------------------------------------------
    # Read actual camera mode
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

    actual_camera_fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or TARGET_FPS
    )

    print(
        f"[CAMERA] "
        f"{actual_width}x{actual_height} "
        f"@ {actual_camera_fps:.1f} FPS"
    )

    # ========================================================
    # WINDOW
    # ========================================================

    window_name = (
        "PrivaStream - buffalo_s"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        window_name,
        on_mouse_click
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    frame_count = 0

    fps_start = time.perf_counter()

    measured_fps = 0.0

    total_ai_ms = 0.0

    total_tracking_ms = 0.0

    total_blur_ms = 0.0

    # ========================================================
    # VIRTUAL CAMERA
    # ========================================================

    with pyvirtualcam.Camera(
        width=actual_width,
        height=actual_height,
        fps=TARGET_FPS,
        fmt=pyvirtualcam.PixelFormat.BGR
    ) as vcam:

        print(
            "[VCAM] Virtual camera started."
        )

        # ====================================================
        # MAIN LOOP
        # ====================================================

        while True:

            frame_start = (
                time.perf_counter()
            )

            # =================================================
            # 1. CAPTURE
            # =================================================

            ret, frame = cap.read()

            if not ret:

                print(
                    "[CAMERA] Failed to read frame."
                )

                break

            # =================================================
            # 2. INSIGHTFACE EVERY FRAME
            # =================================================

            ai_start = (
                time.perf_counter()
            )

            faces = app.get(
                frame
            )

            ai_time_ms = (
                time.perf_counter()
                -
                ai_start
            ) * 1000.0

            total_ai_ms += (
                ai_time_ms
            )

            # =================================================
            # 3. FACE PROCESSING
            # =================================================

            current_detections = []

            for face in faces:

                bbox = np.asarray(
                    face.bbox,
                    dtype=np.float32
                )

                embedding = np.asarray(
                    face.embedding,
                    dtype=np.float32
                )

                whitelisted, similarity = (
                    find_best_whitelist_match(
                        embedding
                    )
                )

                current_detections.append(
                    {
                        "bbox": bbox,
                        "embedding": embedding,
                        "whitelisted": whitelisted,
                        "similarity": similarity
                    }
                )

            # =================================================
            # 4. TRACKING
            # =================================================

            tracking_start = (
                time.perf_counter()
            )

            associate_detections(
                current_detections
            )

            tracking_time_ms = (
                time.perf_counter()
                -
                tracking_start
            ) * 1000.0

            total_tracking_ms += (
                tracking_time_ms
            )

            # =================================================
            # 5. ENROLLMENT
            # =================================================

            process_enrollment()

            # =================================================
            # 6. OUTPUT COPY
            # =================================================

            output_frame = (
                frame.copy()
            )

            # =================================================
            # 7. BLUR EVERY FRAME
            # =================================================

            blur_start = (
                time.perf_counter()
            )

            for track in active_tracks.values():

                if not track.whitelisted:

                    apply_fast_blur(
                        output_frame,
                        track.bbox
                    )

            blur_time_ms = (
                time.perf_counter()
                -
                blur_start
            ) * 1000.0

            total_blur_ms += (
                blur_time_ms
            )

            # =================================================
            # 8. VIRTUAL CAMERA
            # =================================================

            vcam.send(
                output_frame
            )

            # =================================================
            # 9. LOCAL DISPLAY
            # =================================================

            display_frame = (
                output_frame.copy()
            )

            for track in active_tracks.values():

                x1, y1, x2, y2 = [
                    int(v)
                    for v in track.bbox
                ]

                # ------------------------------------------------
                # Clip coordinates
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Status
                # ------------------------------------------------

                if track.whitelisted:

                    color = (
                        0,
                        255,
                        0
                    )

                    status = "PRIMARY"

                else:

                    color = (
                        0,
                        0,
                        255
                    )

                    status = "BLURRED"

                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # ------------------------------------------------
                # Track label
                # ------------------------------------------------

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

            # =================================================
            # 10. PERFORMANCE
            # =================================================

            frame_count += 1

            elapsed = (
                time.perf_counter()
                -
                fps_start
            )

            if elapsed >= 1.0:

                measured_fps = (
                    frame_count
                    /
                    elapsed
                )

                average_ai = (
                    total_ai_ms
                    /
                    frame_count
                )

                average_tracking = (
                    total_tracking_ms
                    /
                    frame_count
                )

                average_blur = (
                    total_blur_ms
                    /
                    frame_count
                )

                print(
                    f"[PERF] "
                    f"FPS={measured_fps:.1f} | "
                    f"AI={average_ai:.1f}ms | "
                    f"Track={average_tracking:.2f}ms | "
                    f"Blur={average_blur:.2f}ms"
                )

                frame_count = 0

                total_ai_ms = 0.0

                total_tracking_ms = 0.0

                total_blur_ms = 0.0

                fps_start = (
                    time.perf_counter()
                )

            # =================================================
            # 11. HUD
            # =================================================

            frame_time_ms = (
                time.perf_counter()
                -
                frame_start
            ) * 1000.0

            hud = [

                f"FPS: "
                f"{measured_fps:.1f}",

                f"Frame time: "
                f"{frame_time_ms:.1f} ms",

                f"AI: "
                f"{ai_time_ms:.1f} ms",

                f"Tracking: "
                f"{tracking_time_ms:.2f} ms",

                f"Blur: "
                f"{blur_time_ms:.2f} ms",

                f"Faces: "
                f"{len(current_detections)}",

                f"Tracks: "
                f"{len(active_tracks)}",

                f"Model: "
                f"{MODEL_NAME}",

                f"Resolution: "
                f"{actual_width}x{actual_height}",

                "MODE: AI EVERY FRAME"
            ]

            y = 25

            for text in hud:

                cv2.putText(
                    display_frame,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                y += 22

            # =================================================
            # 12. DISPLAY
            # =================================================

            cv2.imshow(
                window_name,
                display_frame
            )

            # =================================================
            # 13. KEYBOARD
            # =================================================

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == ord("q"):

                break

            elif key == ord("r"):

                reset_permissions()

            # =================================================
            # 14. 60 FPS PACING
            # =================================================

            vcam.sleep_until_next_frame()

    # ========================================================
    # CLEANUP
    # ========================================================

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