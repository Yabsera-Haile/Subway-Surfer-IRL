import bisect
import collections
import json
import os

import numpy as np

from pipeline import CLASSES, GESTURES, WINDOW_N, resample


def stream(session, model, raw_dir="data/raw", batch=2048):
    """Probabilities for the window ending at every 30 Hz frame."""
    d = np.load(os.path.join(raw_dir, f"{session}.npz"))
    _, grid_t, grid, valid = resample(d["timestamps"], d["landmarks"], d["frame_ok"])
    view = np.lib.stride_tricks.sliding_window_view(grid, WINDOW_N, axis=0)
    ok = np.lib.stride_tricks.sliding_window_view(valid, WINDOW_N).all(axis=1)
    probs = np.full((len(view), len(CLASSES)), np.nan, np.float32)
    idx = np.flatnonzero(ok)
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        probs[b] = model.predict_windows(np.moveaxis(view[b], -1, 1))
    return grid_t[WINDOW_N - 1:], probs


def triggers(times, probs, trigger):
    trigger.reset()
    fires = []
    for t, p in zip(times, probs):
        event = trigger.step(t, None if np.isnan(p[0]) else p)
        if event:
            fires.append((float(t), event))
    return fires


def score(session, fires, labels_dir="data/labels", raw_dir="data/raw"):
    with open(os.path.join(raw_dir, f"{session}_meta.json"), encoding="utf-8") as f:
        cues = json.load(f)["cues"]
    with open(os.path.join(labels_dir, f"{session}.json"), encoding="utf-8") as f:
        reps = json.load(f)["reps"]

    events, ignore = {}, []
    for r in reps:
        c = cues[r["cue"]]
        if r["status"] == "ok":
            events[r["cue"]] = {"label": r["label"], "onset": c["t_cue"] + r["onset"], "latency": None}
        else:
            ignore.append((cues[r["cue"] - 1]["t_end"], c["t_end"]))

    spans = []
    for ci, c in enumerate(cues):
        if c["kind"] == "rest":
            spans.append((c["t_start"], c["t_end"], "rest", ci))
        else:
            spans.append((cues[ci - 1]["t_end"], c["t_cue"], "ready", ci))
            spans.append((c["t_cue"], c["t_end"], c["kind"], ci))
    starts = [s[0] for s in spans]
    t_lo, t_hi = cues[0]["t_start"], cues[-1]["t_end"]

    false = collections.Counter()
    for t, label in fires:
        if not t_lo <= t <= t_hi or any(a <= t <= b for a, b in ignore):
            continue
        _, end, kind, ci = spans[max(bisect.bisect_right(starts, t) - 1, 0)]
        ev = events.get(ci) if kind == "gesture" else None
        if ev and label == ev["label"] and ev["latency"] is None and t >= ev["onset"] - 0.1:
            ev["latency"] = t - ev["onset"]
        elif ev:
            false["wrong class" if label != ev["label"] else "duplicate"] += 1
        else:
            false[kind] += 1

    minutes = (t_hi - t_lo - sum(b - a for a, b in ignore)) / 60
    lat = [e["latency"] for e in events.values() if e["latency"] is not None]
    return {
        "detected": {g: (sum(e["label"] == g and e["latency"] is not None for e in events.values()),
                         sum(e["label"] == g for e in events.values())) for g in GESTURES},
        "false": dict(false),
        "false_per_min": sum(false.values()) / minutes,
        "minutes": minutes,
        "latency_median": float(np.median(lat)) if lat else float("nan"),
        "latency_p90": float(np.percentile(lat, 90)) if lat else float("nan"),
    }


def report(res, indent="  "):
    det = res["detected"]
    hit = sum(a for a, _ in det.values())
    tot = sum(b for _, b in det.values())
    print(f"{indent}detected {hit}/{tot}  " + "  ".join(f"{g} {a}/{b}" for g, (a, b) in det.items())
          + f"   latency median {res['latency_median']:.2f} s, p90 {res['latency_p90']:.2f} s")
    parts = "  ".join(f"{k} {v}" for k, v in sorted(res["false"].items())) or "none"
    print(f"{indent}false triggers {sum(res['false'].values())} ({res['false_per_min']:.1f}/min): {parts}")
