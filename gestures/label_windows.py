import argparse
import collections
import glob
import json
import os

import numpy as np

from pipeline import HZ, MAX_INTERP_S, WINDOW_N, body_centre, resample

WINDOW_S = WINDOW_N / HZ

GESTURE_ENDS_S = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)

REST_SKIP_S = 0.3
REST_STRIDE_S = 0.5
DISTRACTOR_STRIDE_S = 0.25
RECOVERY_FROM_S = 0.4
RECOVERY_STRIDE_S = 0.25

ONSET_RISE = 0.10
ONSET_FLOOR = 0.04
MIN_TRAVEL = 0.20
READY_LIMIT = 0.10
MAX_HOLE_S = 0.15
REST_QUIET = 0.15

EXPECT = {"LEFT": (0, -1), "RIGHT": (0, 1), "JUMP": (1, -1), "DUCK": (1, 1)}


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
        for end in GESTURE_ENDS_S:
            add(label, "gesture", onset + end - WINDOW_S, cue=ci, end=end)
        slide("IDLE", "recovery", onset + RECOVERY_FROM_S, te, RECOVERY_STRIDE_S, cue=ci, after=label)
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
    """Rebuild the labelled windows (X, y, info) from the raw sessions and label files."""
    X, y, info = [], [], []
    for s in sessions:
        with open(os.path.join(labels_dir, f"{s}.json"), encoding="utf-8") as f:
            lab = json.load(f)
        if lab["params"]["HZ"] != HZ or lab["params"]["WINDOW_N"] != WINDOW_N:
            raise SystemExit(f"{s}.json was made with different settings - rerun gestures/label_windows.py")
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


def main():
    p = argparse.ArgumentParser(description="Label raw recording sessions into gesture windows.")
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
