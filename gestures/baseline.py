import argparse
import collections
import glob
import os

import numpy as np

import playback
from label_windows import load
from pipeline import (CLASSES, HZ, TRIGGER_DEFAULTS, Trigger, WindowClassifier, body_centre,
                      mirror, normalize, preset, save_model)
from pose import L_HIP, L_KNEE, L_WRIST, NOSE, R_HIP, R_KNEE, R_WRIST

PRESETS = {
    "balanced": preset(0.9, 2, latency_s=0.133, caught=0.994, false_per_min=0.42),
    "fast": preset(0.9, 1, latency_s=0.100, caught=0.981, false_per_min=0.61),
    "fastest": preset(0.8, 1, latency_s=0.067, caught=0.968, false_per_min=0.89),
}

L2 = 1e-3
ITERS = 3000
LR = 0.5

RECENT = 9
SIGNALS = ["body x", "body y", "arms x", "arms y", "head y", "knee bend", "torso length"]
STATS = ["end", "min", "max", "min velocity", "max velocity",
         "recent change", "recent min velocity", "recent max velocity"]
FEATURE_NAMES = [f"{s} {t}" for s in SIGNALS for t in STATS]


def features(Z):
    ctr, torso = body_centre(Z)
    wr = (Z[:, :, L_WRIST] + Z[:, :, R_WRIST]) / 2
    kn = (Z[:, :, L_KNEE] + Z[:, :, R_KNEE]) / 2
    hp = (Z[:, :, L_HIP] + Z[:, :, R_HIP]) / 2
    signals = [
        ctr[..., 0], ctr[..., 1],
        wr[..., 0] - ctr[..., 0], wr[..., 1] - ctr[..., 1],
        Z[:, :, NOSE, 1] - ctr[..., 1],
        kn[..., 1] - hp[..., 1],
        torso,
    ]
    out = []
    for s in signals:
        s = s - s[:, :1]
        v = np.diff(s, axis=1) * HZ
        r = v[:, -RECENT:]
        out += [s[:, -1], s.min(1), s.max(1), v.min(1), v.max(1),
                s[:, -1] - s[:, -1 - RECENT], r.min(1), r.max(1)]
    return np.stack(out, axis=1).astype(np.float64)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class BaselineClassifier(WindowClassifier):
    type = "baseline"

    def __init__(self, W, b, mu, sd):
        super().__init__()
        self.W, self.b, self.mu, self.sd = (np.asarray(a, np.float64) for a in (W, b, mu, sd))

    @classmethod
    def fit(cls, X, y):
        F = features(normalize(X))
        mu, sd = F.mean(0), F.std(0) + 1e-6
        F = (F - mu) / sd
        Y = (y[:, None] == np.array(CLASSES)[None]).astype(float)
        w = (Y @ (len(y) / (len(CLASSES) * Y.sum(0))))[:, None]
        W, b = np.zeros((F.shape[1], len(CLASSES))), np.zeros(len(CLASSES))
        for _ in range(ITERS):
            P = softmax(F @ W + b)
            G = (P - Y) * w / w.sum()
            W -= LR * (F.T @ G + L2 * W)
            b -= LR * G.sum(0)
        return cls(W, b, mu, sd)

    def predict_windows(self, X):
        return softmax(((features(normalize(X)) - self.mu) / self.sd) @ self.W + self.b)

    def to_json(self):
        return {"features": FEATURE_NAMES,
                "weights": {"W": self.W.tolist(), "b": self.b.tolist(),
                            "mu": self.mu.tolist(), "sd": self.sd.tolist()}}

    @classmethod
    def from_json(cls, doc):
        if doc["features"] != FEATURE_NAMES:
            raise SystemExit("model was trained on a different feature set")
        w = doc["weights"]
        return cls(w["W"], w["b"], w["mu"], w["sd"])


def train_on(sessions):
    X, y, _ = load(sessions)
    Xm, ym = mirror(X, y)
    return BaselineClassifier.fit(np.concatenate([X, Xm]), np.concatenate([y, ym]))


def evaluate(model, session, trigger_params=TRIGGER_DEFAULTS):
    X, y, info = load([session])
    pred = np.array(CLASSES)[model.predict_windows(X).argmax(1)]
    src = np.array([w["source"] for w in info])
    print(f"  windows: accuracy {np.mean(pred == y):.1%}")
    print(f"    {'true / pred':<18}" + "".join(f"{c:>7}" for c in CLASSES))
    for c in CLASSES:
        for so in (["gesture"] if c != "IDLE" else ["rest", "distractor", "recovery"]):
            m = (y == c) & (src == so)
            cnt = collections.Counter(pred[m])
            print(f"    {c + '/' + so:<18}" + "".join(f"{cnt[p]:>7}" for p in CLASSES) + f"   ({m.sum()})")

    times, probs = playback.stream(session, model)
    res = playback.score(session, playback.triggers(times, probs, Trigger(**trigger_params)))
    print("  playback (" + ", ".join(f"{k} {v}" for k, v in trigger_params.items()) + "):")
    playback.report(res, indent="    ")

    print("  threshold trade-off (diagnostic):")
    for th in (0.5, 0.7, 0.8, 0.9):
        r = playback.score(session, playback.triggers(times, probs, Trigger(**{**trigger_params, "thresh": th})))
        hit = sum(a for a, _ in r["detected"].values())
        tot = sum(b for _, b in r["detected"].values())
        print(f"    {th:.1f}: detected {hit}/{tot}, false {r['false_per_min']:.1f}/min")
    return res


def main():
    p = argparse.ArgumentParser(description="Train and evaluate the baseline gesture classifier.")
    p.add_argument("--train", nargs="*")
    p.add_argument("--test", nargs="*")
    p.add_argument("--save", default="models/baseline.json")
    args = p.parse_args()

    labelled = sorted(os.path.basename(f)[:-5] for f in glob.glob("data/labels/*.json"))
    if args.train and args.test:
        splits = [(args.train, s) for s in args.test]
    else:
        if len(labelled) < 2:
            raise SystemExit("need at least two labelled sessions for leave-one-session-out")
        splits = [([t for t in labelled if t != s], s) for s in labelled]

    for train, test in splits:
        print(f"\n=== train {'+'.join(train)} -> test {test} ===")
        evaluate(train_on(train), test, PRESETS["balanced"]["trigger"])

    model = train_on(args.train or labelled)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    save_model(model, args.save, presets=PRESETS)
    print(f"\nsaved model trained on {'+'.join(args.train or labelled)} to {args.save}")


if __name__ == "__main__":
    main()
