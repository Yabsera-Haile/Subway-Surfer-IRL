#!/usr/bin/env python3
"""
STEP 3 - Turn raw sessions into labelled windows.

Run:  python label_windows.py [--sessions s01 s02] [--raw data/raw] [--out data/labels]

Writes data/labels/<session>.json: which stretches of each recording are
LEFT / RIGHT / JUMP / DUCK / IDLE, and every rep that was dropped and why.
Windows are stored as start times, not copied landmarks; load() rebuilds them
from the raw file.
"""

import argparse
import collections
import glob
import json
import os

import numpy as np

from pose import L_HIP, R_HIP, L_SHOULDER, R_SHOULDER

HZ = 30               # sessions are resampled to this so frame rate can't leak into the model
WINDOW_S = 1.0
WINDOW_N = int(round(WINDOW_S * HZ))

# Gesture windows start a little before the movement so the classifier sees
# the transition out of standing, and are repeated at small offsets because a
# live sliding window will catch the gesture anywhere inside it.
PRE_S = 0.2
SHIFTS_S = (-0.2, -0.1, 0.0, 0.1, 0.2)

REST_SKIP_S = 0.3
REST_STRIDE_S = 0.5
DISTRACTOR_STRIDE_S = 0.25
RECOVERY_STRIDE_S = 0.5

ONSET_RISE = 0.10     # body-centre displacement, in torso lengths, that counts as moving
ONSET_FLOOR = 0.04    # walk back from the rise to where the movement actually began
MIN_TRAVEL = 0.20     # a rep that never gets this far the expected way is dropped
READY_LIMIT = 0.10    # moving this much during the countdown means the cue was anticipated
MAX_HOLE_S = 0.15     # a gap this long inside a rep means it was not really observed
MAX_INTERP_S = 0.10   # grid points further than this from real frames are unusable
REST_QUIET = 0.15     # a rest window moving more than this is not idle

EXPECT = {"LEFT": (0, -1), "RIGHT": (0, 1), "JUMP": (1, -1), "DUCK": (1, 1)}  # (axis, sign); y grows down


def resample(ts, lm, ok):
    """Onto a fixed HZ grid anchored at session time 0. Returns (k0, grid_t, grid, valid)."""
    k = np.arange(int(np.ceil(ts[0] * HZ)), int(np.floor(ts[-1] * HZ)) + 1)
    grid_t = k / HZ
    good_t = ts[ok]
    flat = lm[ok].reshape(len(good_t), -1)
    grid = np.stack([np.interp(grid_t, good_t, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    # Interpolating across a hole invents motion that was never seen.
    j = np.clip(np.searchsorted(good_t, grid_t), 1, len(good_t) - 1)
    valid = ((good_t[j] - good_t[j - 1]) <= MAX_INTERP_S) & (grid_t >= good_t[0]) & (grid_t <= good_t[-1])
    return int(k[0]), grid_t, grid.reshape(len(k), *lm.shape[1:]).astype(np.float32), valid


def body_centre(xy):
    sh = (xy[..., L_SHOULDER, :] + xy[..., R_SHOULDER, :]) / 2
    hp = (xy[..., L_HIP, :] + xy[..., R_HIP, :]) / 2
    return (sh + hp) / 2, np.linalg.norm(sh - hp, axis=-1)


def smooth(x):
    return np.convolve(x, np.ones(3) / 3, mode="same")


def label_session(raw_dir, session):
    d = np.load(os.path.join(raw_dir, f"{session}.npz"))
    with open(os.path.join(raw_dir, f"{session}_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    ts, ok = d["timestamps"], d["frame_ok"]
    good_t = ts[ok]
    k0, grid_t, grid, valid = resample(ts, d["landmarks"], ok)
    ctr, torso = body_centre(grid[..., :2])

    windows, reps = [], []
    restless = 0

    def place(t0):
        i0 = int(round(t0 * HZ)) - k0
        if i0 < 0 or i0 + WINDOW_N > len(grid_t) or not valid[i0:i0 + WINDOW_N].all():
            return None
        return i0

    def add(label, source, t0, **extra):
        i0 = place(t0)
        if i0 is not None:
            windows.append({"label": label, "source": source, "t0": round(float(grid_t[i0]), 4), **extra})

    def slide(label, source, t_from, t_to, stride, **extra):
        t = t_from
        while t + WINDOW_S <= t_to + 1e-9:
            add(label, source, t, **extra)
            t += stride

    for ci, c in enumerate(meta["cues"]):
        if c["kind"] == "distractor":
            slide("IDLE", "distractor", c["t_cue"] + 0.3, c["t_end"], DISTRACTOR_STRIDE_S,
                  cue=ci, detail=c["label"])
            continue

        if c["kind"] == "rest":
            t = c["t_start"] + REST_SKIP_S
            while t + WINDOW_S <= c["t_end"] + 1e-9:
                i0 = place(t)
                if i0 is not None:
                    seg = ctr[i0:i0 + WINDOW_N]
                    moved = np.linalg.norm(seg - seg[0], axis=1).max() / np.median(torso[i0:i0 + WINDOW_N])
                    if moved > REST_QUIET:
                        restless += 1
                    else:
                        add("IDLE", "rest", t, cue=ci, lane=c["lane_from"])
                t += REST_STRIDE_S
            continue

        tc, te, label = c["t_cue"], c["t_end"], c["label"]
        rep = {"cue": ci, "label": label}
        reps.append(rep)

        seen = good_t[(good_t >= tc) & (good_t <= te)]
        hole = float(np.diff(seen).max()) if len(seen) > 1 else float("inf")
        # Baseline is the last half second of the rest, before the countdown starts.
        base = (grid_t >= tc - 1.5) & (grid_t < tc - 1.0)
        c0, tl = np.median(ctr[base], axis=0), np.median(torso[base])
        disp = (ctr - c0) / tl
        mag = smooth(np.linalg.norm(disp, axis=1))
        ready = (grid_t >= tc - 1.0) & (grid_t < tc)
        act = np.flatnonzero((grid_t >= tc) & (grid_t <= te))

        rising = act[mag[act] > ONSET_RISE]
        if hole > MAX_HOLE_S:
            reason = f"{hole * 1000:.0f} ms gap during the rep"
        elif mag[ready].max() > READY_LIMIT:
            reason = "moved during the countdown"
        elif not len(rising):
            reason = "no movement"
        else:
            # Walk back only while the signal is still falling: a small drift
            # during the countdown can sit just above the floor for a second,
            # and walking through it would put the onset long before the move.
            j = rising[0]
            while j > act[0] and mag[j] > ONSET_FLOOR and mag[j - 1] < mag[j]:
                j -= 1
            onset = float(grid_t[j])
            axis, sign = EXPECT[label]
            travel = float((sign * disp[act, axis]).max())
            reason = None if travel >= MIN_TRAVEL else f"moved {travel:+.2f} the expected way"

        if reason:
            rep.update(status="dropped", reason=reason)
            continue
        n_before = len(windows)
        for shift in SHIFTS_S:
            add(label, "gesture", onset - PRE_S + shift, cue=ci, shift=shift)
        # What follows a gesture - standing back up from a duck, landing, settling
        # into the new lane - must read as idle, or it will fire gestures of its own.
        slide("IDLE", "recovery", onset - PRE_S + max(SHIFTS_S) + WINDOW_S, te, RECOVERY_STRIDE_S,
              cue=ci, after=label)
        rep.update(status="ok", onset=round(onset - tc, 3), travel=round(travel, 3),
                   windows=sum(w["source"] == "gesture" for w in windows[n_before:]))

    counts = collections.Counter(f"{w['label']}/{w['source']}" for w in windows)
    return {
        "session": session,
        "params": {k: v for k, v in globals().items() if k.isupper() and k != "EXPECT"},
        "summary": {
            "windows": dict(sorted(counts.items())),
            "reps_kept": sum(r["status"] == "ok" for r in reps),
            "reps_dropped": sum(r["status"] == "dropped" for r in reps),
            "restless_rest_windows": restless,
        },
        "reps": reps,
        "windows": windows,
    }


def load(sessions, labels_dir="data/labels", raw_dir="data/raw", sources=None):
    """
    Rebuild windows as arrays: X (n, WINDOW_N, 33, 4) resampled raw landmarks,
    y (n,) labels, info (n,) dicts carrying session/source/cue so splits can be
    made per session and augmented copies of one rep kept together.
    """
    X, y, info = [], [], []
    for s in sessions:
        with open(os.path.join(labels_dir, f"{s}.json"), encoding="utf-8") as f:
            lab = json.load(f)
        if lab["params"]["HZ"] != HZ or lab["params"]["WINDOW_N"] != WINDOW_N:
            raise SystemExit(f"{s}.json was made with different settings - rerun label_windows.py")
        d = np.load(os.path.join(raw_dir, f"{s}.npz"))
        k0, _, grid, _ = resample(d["timestamps"], d["landmarks"], d["frame_ok"])
        for w in lab["windows"]:
            if sources and w["source"] not in sources:
                continue
            i0 = int(round(w["t0"] * HZ)) - k0
            X.append(grid[i0:i0 + WINDOW_N])
            y.append(w["label"])
            info.append({"session": s, **w})
    return np.stack(X), np.array(y), info


def normalize(X):
    """
    (n, T, 33, 4) -> (n, T, 33, 2): x,y relative to where the body was at the
    start of the window, in torso lengths. Relative to the window, not the cue,
    because a live game never knows when a cue happened; relative, not absolute,
    because lane positions drift between sessions.
    """
    xy = X[..., :2]
    ctr, torso = body_centre(xy)
    origin = ctr[:, :3].mean(axis=1)
    scale = np.median(torso, axis=1)
    return (xy - origin[:, None, None]) / scale[:, None, None, None]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/labels")
    p.add_argument("--sessions", nargs="*", help="default: every non-dry session in --raw")
    p.add_argument("--include-dry", action="store_true")
    args = p.parse_args()

    sessions = args.sessions or sorted(
        os.path.basename(f)[:-4] for f in glob.glob(os.path.join(args.raw, "*.npz"))
        if args.include_dry or not os.path.basename(f).startswith("dry_"))
    if not sessions:
        raise SystemExit(f"no sessions found in {args.raw}")
    os.makedirs(args.out, exist_ok=True)

    total = collections.Counter()
    for s in sessions:
        res = label_session(args.raw, s)
        with open(os.path.join(args.out, f"{s}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        sm = res["summary"]
        by_label = collections.Counter()
        for key, n in sm["windows"].items():
            by_label[key.split("/")[0]] += n
            total[key] += n
        idle = {k.split("/")[1]: n for k, n in sm["windows"].items() if k.startswith("IDLE/")}
        print(f"{s}: reps kept {sm['reps_kept']}, dropped {sm['reps_dropped']}")
        for r in res["reps"]:
            if r["status"] == "dropped":
                print(f"      cue #{r['cue']} {r['label']}: {r['reason']}")
        print("      windows  " + "  ".join(f"{k} {by_label[k]}" for k in ("LEFT", "RIGHT", "JUMP", "DUCK", "IDLE"))
              + f"   (idle: {', '.join(f'{k} {v}' for k, v in sorted(idle.items()))})")
        print(f"      restless rest windows skipped: {sm['restless_rest_windows']}")

    if len(sessions) > 1:
        print("\nall sessions: " + "  ".join(f"{k} {v}" for k, v in sorted(total.items())))
    print(f"\nwrote {len(sessions)} file(s) to {args.out}/")


if __name__ == "__main__":
    main()
