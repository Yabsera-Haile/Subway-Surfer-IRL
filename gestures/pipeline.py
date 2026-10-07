import collections
import importlib
import json

import numpy as np

from pose import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER

CLASSES = ["LEFT", "RIGHT", "JUMP", "DUCK", "IDLE"]
GESTURES = CLASSES[:4]
HZ = 30
WINDOW_N = 30
MAX_INTERP_S = 0.10

MODEL_TYPES = {
    "baseline": "baseline:BaselineClassifier",
    "cnn": "deep:CNNClassifier",
    "gru": "deep:GRUClassifier",
}

TRIGGER_DEFAULTS = {
    "thresh": 0.8,
    "hold": 3,
    "release": 0.4,
    "cooldown_s": 0.5,
}


def resample(ts, lm, ok):
    """Batch form of Resampler. Returns (k0, grid_t, grid, valid) for a whole recording."""
    k = np.arange(int(np.ceil(ts[0] * HZ)), int(np.floor(ts[-1] * HZ)) + 1)
    grid_t = k / HZ
    good_t = ts[ok]
    flat = lm[ok].reshape(len(good_t), -1)
    grid = np.stack([np.interp(grid_t, good_t, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    j = np.clip(np.searchsorted(good_t, grid_t), 1, len(good_t) - 1)
    valid = ((good_t[j] - good_t[j - 1]) <= MAX_INTERP_S) & (grid_t >= good_t[0]) & (grid_t <= good_t[-1])
    return int(k[0]), grid_t, grid.reshape(len(k), *lm.shape[1:]).astype(np.float32), valid


class Resampler:
    """Feed raw frames one at a time; get back the 30 Hz frames each one completes."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._t, self._lm, self._k = None, None, None

    def push(self, t, ok, landmarks):
        if not ok:
            return []
        lm = np.asarray(landmarks, np.float64)
        out = []
        if self._t is None:
            self._k = int(np.ceil(t * HZ))
        else:
            valid = (t - self._t) <= MAX_INTERP_S
            while self._k / HZ <= t:
                g = self._k / HZ
                frame = self._lm + (g - self._t) * (lm - self._lm) / (t - self._t)
                out.append((g, frame.astype(np.float32), bool(valid)))
                self._k += 1
        self._t, self._lm = t, lm
        return out


def body_centre(xy):
    sh = (xy[..., L_SHOULDER, :] + xy[..., R_SHOULDER, :]) / 2
    hp = (xy[..., L_HIP, :] + xy[..., R_HIP, :]) / 2
    return (sh + hp) / 2, np.linalg.norm(sh - hp, axis=-1)


def normalize(X):
    """Make window x,y relative to the starting body centre, in torso lengths."""
    xy = X[..., :2]
    ctr, torso = body_centre(xy)
    origin = ctr[:, :3].mean(axis=1)
    scale = np.median(torso, axis=1)
    return (xy - origin[:, None, None]) / scale[:, None, None, None]


def mirror(X, y):
    """Left-right mirror of raw windows (n, T, 33, 4)."""
    pairs = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
             (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
    perm = np.arange(X.shape[2])
    for a, b in pairs:
        perm[a], perm[b] = b, a
    M = X[:, :, perm].copy()
    M[..., 0] = 1 - M[..., 0]
    swap = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
    return M, np.array([swap.get(label, label) for label in y])


class WindowClassifier:
    """Base class for models that classify the last WINDOW_N frames."""

    type = None

    def __init__(self):
        self.reset()

    def reset(self):
        self._buf = collections.deque(maxlen=WINDOW_N)

    def step(self, frame, valid):
        if not valid:
            self._buf.clear()
            return None
        self._buf.append(frame)
        if len(self._buf) < WINDOW_N:
            return None
        return self.predict_windows(np.stack(self._buf)[None])[0]

    def predict_windows(self, X):
        """(n, WINDOW_N, 33, 4) resampled raw windows -> (n, len(CLASSES)) probabilities."""
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    @classmethod
    def from_json(cls, doc):
        raise NotImplementedError


class Trigger:
    """Turn a stream of class probabilities into one event per gesture."""

    def __init__(self, thresh, hold, release, cooldown_s):
        self.thresh, self.hold, self.release, self.cooldown_s = thresh, hold, release, cooldown_s
        self.reset()

    def reset(self):
        self._run = np.zeros(len(GESTURES), int)
        self._last, self._armed = -np.inf, True

    def step(self, t, probs):
        if probs is None:
            self._run[:] = 0
            return None
        g = np.asarray(probs)[:len(GESTURES)]
        self._armed = self._armed or g.max() < self.release
        self._run = np.where(g > self.thresh, self._run + 1, 0)
        if not self._armed or t - self._last < self.cooldown_s:
            return None
        ready = np.flatnonzero(self._run >= self.hold)
        if not len(ready):
            return None
        self._last, self._armed = t, False
        self._run[:] = 0
        return GESTURES[ready[np.argmax(g[ready])]]


def run(frames, model, trigger):
    """Run (t, ok, landmarks) frames through the model and trigger, yielding (t, probs, event)."""
    resampler = Resampler()
    model.reset()
    trigger.reset()
    for t, ok, lm in frames:
        for g, frame, valid in resampler.push(t, ok, lm):
            p = model.step(frame, valid)
            yield g, p, trigger.step(g, p)


def preset(thresh, hold, latency_s, caught, false_per_min):
    """A responsiveness option: trigger settings plus what they measured on held-out sessions."""
    return {"trigger": {**TRIGGER_DEFAULTS, "thresh": thresh, "hold": hold},
            "measured": {"latency_s": latency_s, "caught": caught, "false_per_min": false_per_min}}


def model_doc(model, trigger=None, presets=None):
    presets = presets or {}
    trigger = trigger or presets.get("balanced", {}).get("trigger") or TRIGGER_DEFAULTS
    return {"type": model.type, "version": 1, "classes": CLASSES, "hz": HZ, "window_n": WINDOW_N,
            "trigger": dict(trigger), "presets": presets, **model.to_json()}


def save_model(model, path, trigger=None, presets=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model_doc(model, trigger, presets), f)


def model_from_doc(doc, preset=None):
    for key, want in (("classes", CLASSES), ("hz", HZ), ("window_n", WINDOW_N)):
        if doc[key] != want:
            raise SystemExit(f"model was made with {key}={doc[key]}, this pipeline uses {want}")
    module, cls = MODEL_TYPES[doc["type"]].split(":")
    model = getattr(importlib.import_module(module), cls).from_json(doc)
    params = doc.get("presets", {}).get(preset, {}).get("trigger") or doc["trigger"]
    return model, Trigger(**params)


def load_model(path):
    with open(path, encoding="utf-8") as f:
        return model_from_doc(json.load(f))
