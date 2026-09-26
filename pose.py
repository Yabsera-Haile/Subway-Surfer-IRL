#!/usr/bin/env python3

import collections
import os
import time
import urllib.request

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = "pose_landmarker_lite.task"

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26

NAMED = {
    NOSE: "nose",
    L_SHOULDER: "l_shoulder", R_SHOULDER: "r_shoulder",
    L_ELBOW: "l_elbow", R_ELBOW: "r_elbow",
    L_WRIST: "l_wrist", R_WRIST: "r_wrist",
    L_HIP: "l_hip", R_HIP: "r_hip",
    L_KNEE: "l_knee", R_KNEE: "r_knee",
}

# Skeleton edges to draw. Upper body only 
CONNECTIONS = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (R_HIP, R_KNEE),
]

VIS_THRESHOLD = 0.5  # below this, MediaPipe is guessing at the position

# What the rest of the project imports from here.
__all__ = [
    "MODEL_URL", "MODEL_PATH", "VIS_THRESHOLD",
    "ensure_model", "make_landmarker", "draw_skeleton", "dump_landmarks",
    "NOSE",
    "L_SHOULDER", "R_SHOULDER",
    "L_ELBOW", "R_ELBOW",
    "L_WRIST", "R_WRIST",
    "L_HIP", "R_HIP",
    "L_KNEE", "R_KNEE",
    "NAMED", "CONNECTIONS",
]


def ensure_model():
    """Download the .task file if it isn't here yet. Returns its abspath."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return os.path.abspath(MODEL_PATH)


def make_landmarker(model_path=None):
    
    if model_path is None:
        model_path = ensure_model()

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,  # costs time, we don't need it
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


class Timer:
    """Rolling p50/p95 for each pipeline stage."""

    def __init__(self, n=120):
        self.buf = collections.defaultdict(lambda: collections.deque(maxlen=n))

    def add(self, stage, ms):
        self.buf[stage].append(ms)

    def stats(self, stage):
        d = self.buf[stage]
        if not d:
            return 0.0, 0.0
        a = np.array(d)
        return float(np.percentile(a, 50)), float(np.percentile(a, 95))


def draw_skeleton(frame, lms, w, h):
    for a, b in CONNECTIONS:
        if lms[a].visibility < VIS_THRESHOLD or lms[b].visibility < VIS_THRESHOLD:
            continue
        pa = (int(lms[a].x * w), int(lms[a].y * h))
        pb = (int(lms[b].x * w), int(lms[b].y * h))
        cv2.line(frame, pa, pb, (0, 220, 255), 2)

    for i in NAMED:
        if lms[i].visibility < VIS_THRESHOLD:
            continue
        p = (int(lms[i].x * w), int(lms[i].y * h))
        cv2.circle(frame, p, 4, (0, 80, 255), -1)


def dump_landmarks(lms):
    """Print the raw structure so you can see exactly what you're working with."""
    print("\n--- one frame of landmarks (normalized image coords) ---")
    print(f"{'idx':>4} {'name':<12} {'x':>7} {'y':>7} {'z':>7} {'vis':>6}")
    for i in sorted(NAMED):
        m = lms[i]
        print(f"{i:>4} {NAMED[i]:<12} {m.x:7.3f} {m.y:7.3f} {m.z:7.3f} {m.visibility:6.3f}")
    print("x,y are fractions of frame width/height. y grows DOWNWARD.")
    print("z is depth relative to the hip midpoint - noisy from a single camera.")
    print("vis is MediaPipe's confidence the point is actually visible.\n")


def main():
    model_path = ensure_model()
    landmarker = make_landmarker(model_path)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try VideoCapture(1).")

    timer = Timer()
    t_start = time.perf_counter()
    last_loop = time.perf_counter()
    dump_next = False

    print("Running. Press 'l' to dump landmark values, 'q' to quit.")

    while True:
        loop_t0 = time.perf_counter()

        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        t_capture = (time.perf_counter() - t0) * 1000

        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - t_start) * 1000)  # must increase monotonically
        result = landmarker.detect_for_video(mp_image, ts_ms)
        t_pose = (time.perf_counter() - t0) * 1000

        have_pose = bool(result.pose_landmarks)
        if have_pose:
            lms = result.pose_landmarks[0]
            draw_skeleton(frame, lms, w, h)

            if dump_next:
                dump_landmarks(lms)
                dump_next = False

        
            sx = (lms[L_SHOULDER].x + lms[R_SHOULDER].x) / 2
            sy = (lms[L_SHOULDER].y + lms[R_SHOULDER].y) / 2
            hx = (lms[L_HIP].x + lms[R_HIP].x) / 2
            hy = (lms[L_HIP].y + lms[R_HIP].y) / 2
            torso = float(np.hypot(sx - hx, sy - hy))
            hips_vis = min(lms[L_HIP].visibility, lms[R_HIP].visibility)
        else:
            torso, hips_vis = 0.0, 0.0

        now = time.perf_counter()
        t_loop = (now - last_loop) * 1000
        last_loop = now

        timer.add("capture", t_capture)
        timer.add("pose", t_pose)
        timer.add("loop", t_loop)

        cap50, cap95 = timer.stats("capture")
        pos50, pos95 = timer.stats("pose")
        lp50, lp95 = timer.stats("loop")
        fps = 1000.0 / lp50 if lp50 else 0

        hud = [
            f"FPS {fps:5.1f}   loop p50 {lp50:5.1f}ms  p95 {lp95:5.1f}ms",
            f"capture {cap50:5.1f}ms   pose {pos50:5.1f}ms (p95 {pos95:5.1f})",
            f"pose_found={have_pose}  torso_scale={torso:.3f}  hip_vis={hips_vis:.2f}",
        ]
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(frame, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("step 1 - pose debug", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("l"):
            dump_next = True

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()