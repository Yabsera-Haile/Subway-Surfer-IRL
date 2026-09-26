#!/usr/bin/env python3

import argparse
import collections
import datetime
import json
import os
import random
import sys
import threading
import time
import traceback

import cv2
import numpy as np

import mediapipe as mp

from pose import (
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_SHOULDER, R_SHOULDER,
    draw_skeleton, make_landmarker,
)

GESTURES = ["LEFT", "RIGHT", "JUMP", "DUCK"]

# Near-misses on purpose: things a player does mid-game that must NOT fire a
# gesture. They are the false positives that would ruin gameplay, so they have
# to be in the data. Labelled idle downstream.
DISTRACTORS = [
    "scratch your head", "reach to the side for a cup",
    "adjust your shirt", "look over your shoulder",
    "shift your weight", "stretch your arms up",
    "wave at the camera", "lean in toward the screen",
    "turn and talk to someone", "rub your eyes",
    "cough", "fold your arms",
    "check your watch", "roll your shoulders",
]

REST_RANGE_S = (2.0, 3.5)
READY_S = 1.0
# Measured on the first real session: movement starts ~1.0-1.1 s after the cue
# and peaks as late as 3 s, so a shorter window ends mid-gesture and spills the
# rest of the movement into the following rest, where it reads as idle.
ACTION_S = 3.0
TAIL_S = 3.0          # keep recording past the last cue so it isn't cut off

# Three lanes, starting in the middle, the way the game plays. LEFT and RIGHT
# are only cued when that lane exists, so you never sidestep out of frame.
LANES = 3
START_LANE = 1
LANE_STEP = {"LEFT": -1, "RIGHT": 1}
MAX_SIDE_RUN = 4      # consecutive lane changes before a jump/duck/distractor

FRAMING_WINDOW_S = 2.0
FRAMING_VIS = 0.6     # a joint counts as seen above this visibility
FRAMING_PASS = 0.8    # fraction of recent frames with all four core joints seen
FRAMING_HIP_Y = 0.70  # hips lower than this leave no room to duck in frame
# A dark room makes the camera expose longer, which costs frame rate and blurs
# exactly the fast part of a gesture. This is a floor for "something is wrong",
# not a target - what matters more is that every session runs at a similar rate.
FRAMING_FPS_MIN = 20.0
CORE_JOINTS = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]

CAMERA_GRACE_S = 2.0  # how long a read outage is tolerated before giving up
MAX_TRIES = 1000      # attempts at a cue order satisfying every constraint

N_LANDMARKS = 33
FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = "record_session"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True, help="session id, used for the output filenames")
    p.add_argument("--reps", type=int, default=20, help="reps per gesture class")
    p.add_argument("--distractors", type=int, default=15, help="number of distractor cues")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seed", type=int, default=None,
                   help="fixes cue order and rest durations; default is random")
    p.add_argument("--dry-run", action="store_true",
                   help="2 reps + 2 distractors, output files prefixed dry_")
    args = p.parse_args()
    if args.dry_run:
        args.reps, args.distractors = 2, 2
    return args


def _run_key(cue):
    # Every distractor shares one key, so a single "no key three times running"
    # rule enforces both constraints at once: no gesture class 3x in a row and
    # no more than 2 distractors in a row.
    return cue[1] if cue[0] == "gesture" else cue[0]


def _completable(remaining, lane):
    """Can the sidesteps still left be spent without walking out of the lanes?"""
    nl, nr = remaining["LEFT"], remaining["RIGHT"]
    if nl and nr:
        return True   # they can always alternate
    if nl:
        return nl <= lane
    if nr:
        return nr <= LANES - 1 - lane
    return True


def _shuffle_no_runs(keys, rng):
    # SIDE gets a longer allowance than the others: its directions are assigned
    # afterwards and alternate, so a run of them is legal, but an unbounded run
    # is a slog to perform and starves the session of jumps and ducks.
    seq = keys[:]
    rng.shuffle(seq)
    for _ in range(10 * MAX_TRIES):
        bad = [i for i in range(2, len(seq))
               if seq[i] == seq[i - 1] == seq[i - 2] != "SIDE"]
        bad += [i for i in range(MAX_SIDE_RUN, len(seq))
                if all(k == "SIDE" for k in seq[i - MAX_SIDE_RUN:i + 1])]
        if not bad:
            return seq
        i, j = rng.choice(bad), rng.randrange(len(seq))
        seq[i], seq[j] = seq[j], seq[i]
    raise RuntimeError("could not satisfy the cue adjacency constraint")


def _assign_directions(slots, rng):
    """Turn each SIDE slot into LEFT or RIGHT. None if the walk corners itself."""
    nl = nr = slots.count("SIDE") // 2
    lane, out = START_LANE, []
    for key in slots:
        if key != "SIDE":
            out.append((key, lane, lane))
            continue
        options = []
        for d, n in (("LEFT", nl), ("RIGHT", nr)):
            nxt = lane + LANE_STEP[d]
            if not n or not 0 <= nxt < LANES:
                continue
            left = {"LEFT": nl - (d == "LEFT"), "RIGHT": nr - (d == "RIGHT")}
            if _completable(left, nxt):
                options.append(d)
        if not options:
            return None
        d = rng.choices(options, weights=[nl if x == "LEFT" else nr for x in options])[0]
        nxt = lane + LANE_STEP[d]
        nl, nr = nl - (d == "LEFT"), nr - (d == "RIGHT")
        out.append((d, lane, nxt))
        lane = nxt
    return out


def _distractor_labels(n, rng):
    # Without replacement until the pool is exhausted, then reshuffled, so no
    # distractor repeats before every other one has been used.
    out = []
    while len(out) < n:
        pool = DISTRACTORS[:]
        rng.shuffle(pool)
        out.extend(pool)
    return out[:n]


def build_cues(reps, n_distractors, rng):
    # Distractors are interleaved with the gestures rather than run as a
    # trailing block: a block lets slow session drift (fatigue, shifting
    # stance, light) line up with the idle class, and the classifier would
    # learn the drift instead of the pose.
    # Sidesteps are placed as direction-less SIDE slots and only turned into
    # LEFT/RIGHT afterwards. Choosing the direction at placement time made the
    # lane rule defer sidesteps whenever the walk sat against an edge, and they
    # then piled up at the end of the session.
    keys = (["SIDE"] * (2 * reps) + ["JUMP"] * reps + ["DUCK"] * reps
            + ["distractor"] * n_distractors)
    worst = max(reps, n_distractors)
    if worst > 2 * (len(keys) - worst + 1):
        raise SystemExit(f"cannot place {worst} cues of one kind among {len(keys)} "
                         "without three in a row - raise --reps or lower --distractors")

    for _ in range(MAX_TRIES):
        walk = _assign_directions(_shuffle_no_runs(keys, rng), rng)
        if walk is not None:
            labels = iter(_distractor_labels(n_distractors, rng))
            return [(("distractor", next(labels)) if k == "distractor" else ("gesture", k)) + (a, b)
                    for k, a, b in walk]
    raise RuntimeError(f"no legal cue order found in {MAX_TRIES} attempts")


def log_entry(kind, label, t_start, lane_from, lane_to):
    """One shape for every cue-log entry, null where a field doesn't apply."""
    return {"kind": kind, "label": label, "t_start": t_start, "t_cue": None, "t_end": None,
            "lane_from": lane_from, "lane_to": lane_to}


class CueScheduler:
    """
    Walks each cue through rest -> ready -> cue, clocked entirely by the
    caller. The capture loop calls update(now) once per frame and the
    scheduler just compares against the clock, so nothing ever sleeps and
    the loop never misses a frame. Sleeping would tear a hole in the
    recording exactly where the gesture is.
    """

    def __init__(self, cues, rng):
        self.cues = cues
        self.rng = rng
        self.state = "framing"
        self.i = -1
        self.next_t = None
        self.log = []
        self._pending = None

    @property
    def label(self):
        return self.cues[self.i][1]

    @property
    def kind(self):
        return self.cues[self.i][0]

    @property
    def lane_from(self):
        return self.cues[self.i][2]

    @property
    def lane_to(self):
        return self.cues[self.i][3]

    def start(self, now):
        self._next_cue(now)

    def _next_cue(self, now):
        self.i += 1
        if self.i == len(self.cues):
            # Keep rolling past the last cue: movement starts about a second
            # after it and would otherwise be cut off mid-gesture.
            self.state, self.next_t = "tail", now + TAIL_S
            return
        self.state = "rest"
        self.next_t = now + self.rng.uniform(*REST_RANGE_S)
        self._pending = log_entry("rest", None, now, self.lane_from, self.lane_from)

    def update(self, now):
        """Advance past an elapsed phase boundary. True on the frame that must first show the cue."""
        if self.next_t is None or now < self.next_t:
            return False
        if self.state == "rest":
            self._pending["t_end"] = now
            self.log.append(self._pending)
            self.state, self.next_t = "ready", now + READY_S
        elif self.state == "ready":
            self._pending = log_entry(self.kind, self.label, now, self.lane_from, self.lane_to)
            self.state, self.next_t = "cue", None   # armed by cue_shown()
            return True
        elif self.state == "cue":
            self._pending["t_end"] = now
            self.log.append(self._pending)
            self._next_cue(now)
        elif self.state == "tail":
            self.state, self.next_t = "done", None
        return False

    def cue_shown(self, t):
        self._pending["t_cue"] = t
        self.next_t = t + ACTION_S

    def countdown(self, now):
        return max(1, int(np.ceil(self.next_t - now)))


def make_beeper():
    """
    Audio must never stall the capture loop or kill a session: the beep runs
    on its own thread and any failure downgrades to the terminal bell. One
    output stream is opened up front because sd.play() opens a fresh stream
    per call, which on Windows lands the beep 0.5-1 s after the cue.
    """
    stream = None
    try:
        import sounddevice as sd
        sr = 44100
        t = np.arange(int(sr * 0.12)) / sr
        tone = 0.5 * np.sin(2 * np.pi * 880 * t)
        fade = int(sr * 0.005)
        ramp = np.linspace(0.0, 1.0, fade)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]
        tone = tone.astype(np.float32)
        stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", latency="low")
        stream.start()
    except Exception as e:
        print(f"audio: using terminal bell ({e})")
        stream = None

    def play():
        nonlocal stream
        if stream is not None:
            try:
                stream.write(tone)
                return
            except Exception as e:
                print(f"audio: using terminal bell from now on ({e})")
                stream = None
        print("\a", end="", flush=True)

    def beep():
        threading.Thread(target=play, daemon=True).start()

    return beep


def put_text(frame, text, org, scale, color, thick):
    # Outline by stamping black copies around the text rather than a thicker
    # pass underneath: OpenCV 5 maps thickness to font weight, which changes
    # the glyph advance, so a thicker pass no longer lines up with the fill.
    x, y = org
    k = thick + 1
    for dx, dy in ((-k, 0), (k, 0), (0, -k), (0, k), (-k, -k), (k, k), (-k, k), (k, -k)):
        cv2.putText(frame, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def wrap_to_width(text, scale, thick, max_w):
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if cur and cv2.getTextSize(cand, FONT, scale, thick)[0][0] > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    return lines + [cur]


def draw_big(frame, text, scale, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    thick = 3
    lines = wrap_to_width(text, scale, thick, w - 40)
    (_, th), base = cv2.getTextSize("A", FONT, scale, thick)
    step = int((th + base) * 1.15)
    y = (h - step * len(lines)) // 2 + th
    for line in lines:
        tw = cv2.getTextSize(line, FONT, scale, thick)[0][0]
        put_text(frame, line, ((w - tw) // 2, y), scale, color, thick)
        y += step


def framing_score(window):
    vis = np.nan_to_num(np.array([v for _, v, _ in window], dtype=np.float32))
    hip_y = np.nan_to_num(np.array([y for _, _, y in window], dtype=np.float32))
    return float(np.mean(np.all(vis > FRAMING_VIS, axis=1) & (hip_y < FRAMING_HIP_Y)))


def framing_fps(window):
    span = window[-1][0] - window[0][0] if len(window) > 1 else 0.0
    return (len(window) - 1) / span if span > 0 else 0.0


def framing_diagnosis(window):
    vis = np.array([v for _, v, _ in window], dtype=np.float32)
    hip_y = np.array([y for _, _, y in window], dtype=np.float32)
    if len(vis) == 0 or np.isnan(vis[:, 0]).mean() > 0.5:
        return "no pose detected - step into frame"
    seen = dict(zip(CORE_JOINTS, (np.nan_to_num(vis) > FRAMING_VIS).mean(axis=0)))
    if min(seen[L_HIP], seen[R_HIP]) < FRAMING_PASS:
        return "hips not in frame - step back"
    if min(seen[L_SHOULDER], seen[R_SHOULDER]) < FRAMING_PASS:
        return "shoulders not visible - centre yourself"
    # Hips near the bottom edge means a duck pushes them out of frame, and
    # MediaPipe then extrapolates them instead of seeing them.
    if np.nanmean(hip_y > FRAMING_HIP_Y) > 1 - FRAMING_PASS:
        return "no room to duck - tilt the camera down or step back"
    fps = framing_fps(window)
    if fps < FRAMING_FPS_MIN:
        return f"only {fps:.0f} fps - more light on you, the camera is exposing too long"
    return "framing unstable - hold still, face the camera"


def draw_lanes(frame, lane):
    w = frame.shape[1]
    box, gap = w // 9, w // 40
    x0 = (w - (LANES * box + (LANES - 1) * gap)) // 2
    for i in range(LANES):
        x = x0 + i * (box + gap)
        on = i == lane
        cv2.rectangle(frame, (x, 40), (x + box, 40 + box // 2),
                      (0, 220, 0) if on else (90, 90, 90), -1 if on else 2)


def open_camera(index):
    # DirectShow first: on the machine this was built for, Media Foundation
    # (OpenCV's Windows default) ran the full capture+pose+display loop at
    # 21 fps where DirectShow held 30. The default is only a fallback.
    for api, name in ((cv2.CAP_DSHOW, "dshow"), (cv2.CAP_ANY, "default")):
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            return cap, name
        cap.release()
    return None, None


def toggle_fullscreen():
    full = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def fps_stats(ts):
    if len(ts) < 2:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "n_frames": int(len(ts))}
    inst = 1.0 / np.diff(ts)
    return {
        "mean": round(float((len(ts) - 1) / (ts[-1] - ts[0])), 1),
        "p50": round(float(np.percentile(inst, 50)), 1),
        "p95": round(float(np.percentile(inst, 95)), 1),
        "n_frames": int(len(ts)),
    }


def stage_stats(stage_ms):
    if not stage_ms:
        return {}
    a = np.array(stage_ms)
    return {f"{name}_{q}": round(float(np.percentile(a[:, i], int(q[1:]))), 1)
            for i, name in enumerate(("capture", "pose")) for q in ("p50", "p95")}


def write_outputs(out_dir, stem, lm, ts, ok, meta):
    npz_path = os.path.join(out_dir, f"{stem}.npz")
    meta_path = os.path.join(out_dir, f"{stem}_meta.json")
    np.savez(npz_path, landmarks=lm, timestamps=ts, frame_ok=ok)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return npz_path, meta_path


def report(lm, ts, ok, cues, reps, framing, aborted, stage=None):
    """Print the end-of-session summary. Returns the list of hard problems found."""
    problems = []
    print("\n=== session report" + (f"  (ABORTED: {aborted})" if aborted else "") + " ===")

    n = len(ts)
    if n < 2:
        problems.append("fewer than 2 frames recorded")
        print(f"  PROBLEM: {problems[0]}")
        return problems
    if not (lm.shape[0] == n == ok.shape[0]):
        problems.append(f"array lengths differ: landmarks {lm.shape[0]}, timestamps {n}, frame_ok {ok.shape[0]}")
    if not np.all(np.diff(ts) > 0):
        problems.append("timestamps are not strictly increasing")

    dt = np.diff(ts) * 1000
    print(f"  frames {n}   duration {ts[-1] - ts[0]:.1f} s   "
          f"loop ms mean {dt.mean():.1f}  p50 {np.percentile(dt, 50):.1f}  p95 {np.percentile(dt, 95):.1f}")

    if stage:
        print(f"  of which  capture p50 {stage['capture_p50']:.1f} ms (p95 {stage['capture_p95']:.1f})   "
              f"pose p50 {stage['pose_p50']:.1f} ms (p95 {stage['pose_p95']:.1f})")
        if stage["capture_p50"] > 40:
            print("  WARNING: the camera is pacing the loop - more light, or lower the requested resolution")
        elif stage["pose_p50"] > 25:
            print("  WARNING: pose inference is pacing the loop - close other apps or use a smaller model")

    bad = ~ok
    print(f"  no-pose frames {int(bad.sum())} ({100 * bad.mean():.1f}%)   longest run {longest_run(bad)}")

    counts = collections.Counter(c["label"] for c in cues if c["kind"] == "gesture")
    n_dis = sum(c["kind"] == "distractor" for c in cues)
    print("  cues  " + "  ".join(f"{g} {counts[g]}" for g in GESTURES) + f"  distractor {n_dis}")
    if not aborted:
        for g in GESTURES:
            if counts[g] != reps:
                problems.append(f"{g}: {counts[g]} cues logged, expected {reps}")

    hip = np.fmin(lm[:, L_HIP, 3], lm[:, R_HIP, 3]) > 0.5
    knee = np.fmin(lm[:, L_KNEE, 3], lm[:, R_KNEE, 3]) > 0.5
    print(f"  hip vis > 0.5 on {100 * hip.mean():.1f}% of frames   "
          f"knee vis > 0.5 on {100 * knee.mean():.1f}% (decides whether knees stay in the feature set)")
    if hip.mean() < 0.9:
        print("  WARNING: hips visible on under 90% of frames - fix framing before recording more")

    if framing and cues:
        idle = ts[ts < framing["t_end"]]
        active = ts[(ts >= cues[0]["t_start"]) & (ts <= cues[-1]["t_end"])]
        if len(idle) > 1 and len(active) > 1:
            fps_idle = (len(idle) - 1) / (idle[-1] - idle[0])
            fps_active = (len(active) - 1) / (active[-1] - active[0])
            print(f"  fps  framing {fps_idle:.1f}   cued {fps_active:.1f}")
            if fps_active < 0.85 * fps_idle:
                problems.append(f"cued fps {fps_active:.1f} is more than 15% below framing fps "
                                f"{fps_idle:.1f} - something is blocking the capture loop")

    for p in problems:
        print(f"  PROBLEM: {p}")
    return problems


def main():
    args = parse_args()
    stem = ("dry_" if args.dry_run else "") + args.session
    os.makedirs(args.out, exist_ok=True)
    if not args.dry_run and os.path.exists(os.path.join(args.out, f"{stem}.npz")):
        raise SystemExit(f"{args.out}/{stem}.npz already exists - pick another --session")

    seed = random.randrange(2 ** 31) if args.seed is None else args.seed
    rng = random.Random(seed)
    cues = build_cues(args.reps, args.distractors, rng)
    sched = CueScheduler(cues, rng)
    beep = make_beeper()
    landmarker = make_landmarker()

    cap, backend = open_camera(args.camera)
    if cap is None:
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    camera = {"index": args.camera, "backend": backend,
              "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}

    # The user reads this from 2-3 m: fullscreen by default, with a sensible
    # windowed size to fall back to (OpenCV's Win32 default is a tiny window).
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 960)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"session {args.session}: {len(cues)} cues "
          f"({args.reps} x {len(GESTURES)} gestures + {args.distractors} distractors), seed {seed}")
    print("Stand in the middle lane, far enough back that your hips sit above the lower "
          "third of the frame. SPACE starts once the number is green. Move only when the "
          "cue appears. q aborts - everything so far is still saved. f toggles fullscreen.")

    frames = []   # (t, landmark row, pose found) - one tuple so the arrays can never drift apart
    stage_ms = []
    nan_row = np.full((N_LANDMARKS, 4), np.nan, dtype=np.float32)
    h, w = camera["height"] or 480, camera["width"] or 640
    framing_win = collections.deque()
    framing_meta = None
    score, fps_now, ready = 0.0, 0.0, False
    refuse_msg, refuse_until = "", 0.0
    aborted = None
    last_ms = -1
    last_good = 0.0
    started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    t0 = time.perf_counter()

    def clock():
        return time.perf_counter() - t0

    def pump(img):
        """Paint a frame and take the keys that work in any state. Returns (abort reason, key)."""
        cv2.imshow(WIN, img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("f"):
            toggle_fullscreen()
        if k == ord("q"):
            return "q pressed", k
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            return "window closed", k
        return None, k

    try:
        while sched.state != "done":
            read_t0 = time.perf_counter()
            ok, frame = cap.read()
            t = clock()
            read_ms = (time.perf_counter() - read_t0) * 1000
            if not ok:
                # Windows webcams occasionally hand back one bad read; only a
                # sustained outage should end a 20-minute session.
                if t - last_good > CAMERA_GRACE_S:
                    aborted = "camera stopped delivering frames"
                    break
                # Still paint and read keys: otherwise the view freezes, q
                # stops working for the whole grace window, and a dead camera
                # spins this loop as fast as it can return failures.
                dropout = np.zeros((h, w, 3), np.uint8)
                draw_big(dropout, "camera dropped out", 2.0, (0, 0, 255))
                aborted, _ = pump(dropout)
                if aborted:
                    break
                continue
            last_good = t
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # MediaPipe rejects a timestamp that doesn't move forward, and two
            # frames can land in the same millisecond - nudge rather than drop.
            ms = max(int(t * 1000), last_ms + 1)
            last_ms = ms
            pose_t0 = time.perf_counter()
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
            # Split the loop time so a slow session says whether the camera is
            # pacing it (long read) or the model is (long pose).
            stage_ms.append((read_ms, (time.perf_counter() - pose_t0) * 1000))

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                row = np.array([[m.x, m.y, m.z, m.visibility] for m in lms], dtype=np.float32)
                draw_skeleton(frame, lms, w, h)
            else:
                lms, row = None, nan_row
            # A frame with no pose still gets a row (all NaN) so the arrays stay
            # index-aligned with timestamps. Skipping it would shift every
            # downstream time lookup.
            frames.append((t, row, lms is not None))

            show_cue = False
            if sched.state == "framing":
                framing_win.append((t, row[CORE_JOINTS, 3], (row[L_HIP, 1] + row[R_HIP, 1]) / 2))
                while framing_win[0][0] < t - FRAMING_WINDOW_S:
                    framing_win.popleft()
                score = framing_score(framing_win)
                fps_now = framing_fps(framing_win)
                ready = score >= FRAMING_PASS and fps_now >= FRAMING_FPS_MIN
                if t < refuse_until:
                    draw_big(frame, refuse_msg, 1.5, (0, 0, 255))
                else:
                    draw_big(frame, f"{int(score * 100)}%", 3.0, (0, 220, 0) if ready else (0, 0, 255))
                    if fps_now and fps_now < FRAMING_FPS_MIN:
                        msg = f"{fps_now:.0f} fps - too dark"
                        tw = cv2.getTextSize(msg, FONT, 1.2, 3)[0][0]
                        put_text(frame, msg, ((w - tw) // 2, h - 55), 1.2, (0, 0, 255), 3)
                status = "framing check - SPACE to start"
            elif sched.state == "tail":
                sched.update(t)
                draw_big(frame, "hold still", 2.0)
                status = "finishing"
            else:
                show_cue = sched.update(t)
                if sched.state == "ready":
                    draw_big(frame, str(sched.countdown(t)), 5.0)
                elif sched.state == "cue":
                    draw_big(frame, sched.label, 4.0 if sched.kind == "gesture" else 3.0, (0, 255, 255))
                if sched.state in ("rest", "ready", "cue"):
                    # during the cue, highlight where you should end up
                    draw_lanes(frame, sched.lane_to if sched.state == "cue" else sched.lane_from)
                status = f"cue {min(sched.i + 1, len(cues))} / {len(cues)}   {sched.state}"

            put_text(frame, f"{args.session}   q abort   f fullscreen", (10, 22), 0.6, (255, 255, 255), 1)
            put_text(frame, status, (10, h - 12), 0.6, (255, 255, 255), 1)

            if show_cue:
                beep()
            aborted, key = pump(frame)
            if show_cue:
                # waitKey is what actually paints the window, so this is the
                # first instant the cue is visible - the action window runs from here.
                sched.cue_shown(clock())
            if aborted:
                break

            if key == ord(" ") and sched.state == "framing":
                if ready:
                    framing_meta = {"t_end": t, "score": round(score, 3), "fps": round(fps_now, 1)}
                    sched.start(t)
                    print(f"framing {int(score * 100)}% at {fps_now:.0f} fps - recording")
                else:
                    refuse_msg = framing_diagnosis(framing_win)
                    refuse_until = t + 3.0
                    print(f"framing {int(score * 100)}% at {fps_now:.0f} fps "
                          f"- not starting: {refuse_msg}")
    except KeyboardInterrupt:
        aborted = "interrupted"
    except Exception as e:
        aborted = f"crashed: {e!r}"
        traceback.print_exc()

    # Write before releasing anything: camera teardown can take a while and a
    # second Ctrl-C there must not cost the recording.
    ts = np.array([f[0] for f in frames], dtype=np.float64)
    lm = np.stack([f[1] for f in frames]) if frames else np.zeros((0, N_LANDMARKS, 4), dtype=np.float32)
    ok = np.array([f[2] for f in frames], dtype=bool)

    meta = {
        "session": args.session,
        "started_at": started_at,
        "camera": camera,
        "fps": fps_stats(ts),
        "stage_ms": stage_stats(stage_ms),
        "reps_per_class": args.reps,
        "distractors": args.distractors,
        "seed": seed,
        "lanes": LANES,
        "action_s": ACTION_S,
        "tail_s": TAIL_S,
        "dry_run": args.dry_run,
        "framing": framing_meta,
        "aborted": aborted,
        "cues": sched.log,
        "notes": "",
    }
    npz_path, meta_path = write_outputs(args.out, stem, lm, ts, ok, meta)
    print(f"\nwrote {npz_path}\nwrote {meta_path}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if report(lm, ts, ok, sched.log, args.reps, framing_meta, aborted, meta["stage_ms"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
