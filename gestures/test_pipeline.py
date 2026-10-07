import argparse
import glob
import json
import os
import sys

import numpy as np

import playback
from pipeline import CLASSES, GESTURES, load_model, model_from_doc, run

TOL = 1e-4


def raw_frames(session, t_from=-np.inf, t_to=np.inf):
    d = np.load(f"data/raw/{session}.npz")
    keep = (d["timestamps"] >= t_from) & (d["timestamps"] <= t_to)
    return list(zip(d["timestamps"][keep].tolist(), d["frame_ok"][keep].tolist(), d["landmarks"][keep]))


def check_stream_matches_batch(model, trigger, session):
    times, probs = playback.stream(session, model)
    batch = {round(t, 6): p for t, p in zip(times, probs)}
    batch_events = playback.triggers(times, probs, trigger)

    worst, compared, mismatched = 0.0, 0, 0
    live_events = []
    for t, p, event in run(raw_frames(session), model, trigger):
        b = batch.get(round(t, 6))
        if b is None:
            continue
        if p is None or np.isnan(b[0]):
            mismatched += (p is None) != bool(np.isnan(b[0]))
            continue
        worst = max(worst, float(np.abs(p - b).max()))
        compared += 1
        if event:
            live_events.append((t, event))

    same_events = [e for _, e in live_events] == [e for _, e in batch_events] and \
        all(abs(a - b) < 1e-6 for (a, _), (b, _) in zip(live_events, batch_events))
    ok = worst < TOL and mismatched == 0 and same_events
    print(f"  {session}: {compared} frames compared, max |prob diff| {worst:.2e}, "
          f"valid/invalid disagreements {mismatched}, events {len(live_events)} vs {len(batch_events)} "
          f"{'identical' if same_events else 'DIFFER'}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def golden_segment(session):
    """The shortest stretch of a session that contains every gesture, padded either side."""
    with open(f"data/raw/{session}_meta.json", encoding="utf-8") as f:
        cues = [c for c in json.load(f)["cues"] if c["kind"] != "rest"]
    best = None
    for i in range(len(cues)):
        seen = set()
        for j in range(i, len(cues)):
            if cues[j]["kind"] == "gesture":
                seen.add(cues[j]["label"])
            if seen == set(GESTURES):
                span = (cues[i]["t_cue"] - 2.5, cues[j]["t_end"] + 1.0)
                if best is None or span[1] - span[0] < best[1] - best[0]:
                    best = span
                break
    return best


def export_golden(doc, session, path):
    model, trigger = model_from_doc(doc)
    t_from, t_to = golden_segment(session)
    frames = [(round(t, 6), ok, np.round(lm, 6).astype(np.float32)) for t, ok, lm in raw_frames(session, t_from, t_to)]
    out = [{"t": round(t, 6), "probs": None if p is None else [round(float(x), 7) for x in p], "event": e}
           for t, p, e in run(frames, model, trigger)]
    golden = {
        "about": "Raw frames in, 30 Hz probabilities and events out. Any implementation of the "
                 "pipeline must reproduce 'expected' from 'input' to within 'tolerance'.",
        "tolerance": TOL,
        "classes": CLASSES,
        "source": {"session": session, "t_from": round(t_from, 3), "t_to": round(t_to, 3)},
        "model": doc,
        "input": [{"t": t, "ok": ok, "landmarks": lm.astype(np.float64).round(6).tolist() if ok else None}
                  for t, ok, lm in frames],
        "expected": out,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(golden, f, separators=(",", ":"))
    events = [(o["t"], o["event"]) for o in out if o["event"]]
    print(f"  wrote {path}: {len(frames)} raw frames ({t_to - t_from:.1f} s of {session}), "
          f"{len(out)} output frames, events {[(round(t, 2), e) for t, e in events]}, "
          f"{os.path.getsize(path) / 1e6:.1f} MB")


def check_golden(path):
    with open(path, encoding="utf-8") as f:
        golden = json.load(f)
    model, trigger = model_from_doc(golden["model"])
    frames = [(x["t"], x["ok"], np.array(x["landmarks"] or np.full((33, 4), np.nan), np.float32))
              for x in golden["input"]]
    got = list(run(frames, model, trigger))
    want = golden["expected"]
    worst, problems = 0.0, []
    if len(got) != len(want):
        problems.append(f"{len(got)} output frames, expected {len(want)}")
    for (t, p, e), w in zip(got, want):
        if abs(t - w["t"]) > 1e-6:
            problems.append(f"frame time {t} vs {w['t']}")
            break
        if (p is None) != (w["probs"] is None):
            problems.append(f"t={t}: probabilities present/absent mismatch")
        elif p is not None:
            worst = max(worst, float(np.abs(p - np.array(w["probs"])).max()))
        if e != w["event"]:
            problems.append(f"t={t}: event {e} vs {w['event']}")
    ok = not problems and worst < golden["tolerance"]
    print(f"  {os.path.basename(path)}: {len(got)} frames, max |prob diff| {worst:.2e}"
          + (f", problems: {problems[:3]}" if problems else "") + f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Check the pipeline frame-by-frame against batch mode and the golden files.")
    p.add_argument("--model", default="models/baseline.json")
    p.add_argument("--export", action="store_true")
    p.add_argument("--golden-session", default="s02")
    args = p.parse_args()

    model, trigger = load_model(args.model)
    with open(args.model, encoding="utf-8") as f:
        doc = json.load(f)
    golden_path = f"tests/golden_{doc['type']}.json"

    results = []
    print("frame-by-frame vs batch:")
    for path in sorted(glob.glob("data/labels/*.json")):
        results.append(check_stream_matches_batch(model, trigger, os.path.basename(path)[:-5]))

    print("golden file:")
    if args.export:
        export_golden(doc, args.golden_session, golden_path)
    if os.path.exists(golden_path):
        results.append(check_golden(golden_path))
    else:
        print(f"  {golden_path} missing - run with --export")
        results.append(False)

    print("ALL PASS" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
