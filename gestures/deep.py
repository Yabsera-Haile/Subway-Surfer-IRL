import argparse
import glob
import os

import numpy as np

from baseline import evaluate, softmax
from label_windows import load
from pipeline import CLASSES, WindowClassifier, mirror, normalize, preset, save_model

PRESETS = {
    "cnn": {
        "balanced": preset(0.9, 3, latency_s=0.133, caught=0.977, false_per_min=0.59),
        "fast": preset(0.8, 3, latency_s=0.100, caught=0.970, false_per_min=0.76),
        "fastest": preset(0.8, 2, latency_s=0.067, caught=0.964, false_per_min=0.94),
    },
    "gru": {
        "balanced": preset(0.9, 3, latency_s=0.111, caught=0.968, false_per_min=0.56),
        "fast": preset(0.8, 3, latency_s=0.100, caught=0.964, false_per_min=0.80),
        "fastest": preset(0.8, 2, latency_s=0.067, caught=0.964, false_per_min=0.90),
    },
}

JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
RECENT = 9
HIDDEN = 32
EPOCHS = 40
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
NOISE = 0.05


def inputs(X):
    """Build the neural models' input: 13 joints' positions and velocities per frame."""
    Z = normalize(X)[:, :, JOINTS].reshape(len(X), X.shape[1], -1)
    v = np.diff(Z, axis=1, prepend=Z[:, :1])
    return np.concatenate([Z, v], axis=2).astype(np.float32)


def conv1d(x, W, b):
    """1D convolution with 'same' padding; W is laid out (Cout, Cin, K) like PyTorch."""
    k = W.shape[2]
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
    out = sum(xp[:, i:i + x.shape[1]] @ W[:, :, i].T for i in range(k))
    return out + b


def cnn_forward(F, p):
    h = np.maximum(conv1d(F, p["conv1.weight"], p["conv1.bias"]), 0)
    h = np.maximum(conv1d(h, p["conv2.weight"], p["conv2.bias"]), 0)
    pooled = np.concatenate([h.max(axis=1), h[:, -RECENT:].mean(axis=1)], axis=1)
    return pooled @ p["fc.weight"].T + p["fc.bias"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gru_forward(F, p):
    Wi, Wh, bi, bh = p["gru.weight_ih_l0"], p["gru.weight_hh_l0"], p["gru.bias_ih_l0"], p["gru.bias_hh_l0"]
    H = Wh.shape[1]
    h = np.zeros((len(F), H), np.float32)
    for t in range(F.shape[1]):
        gi = F[:, t] @ Wi.T + bi
        gh = h @ Wh.T + bh
        r = sigmoid(gi[:, :H] + gh[:, :H])
        z = sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h = (1 - z) * n + z * h
    return h @ p["fc.weight"].T + p["fc.bias"]


def torch_net(kind, n_in):
    import torch
    from torch import nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(n_in, HIDDEN, 5, padding=2)
            self.conv2 = nn.Conv1d(HIDDEN, HIDDEN, 5, padding=2)
            self.fc = nn.Linear(2 * HIDDEN, len(CLASSES))

        def forward(self, x):
            h = torch.relu(self.conv1(x.transpose(1, 2)))
            h = torch.relu(self.conv2(h))
            pooled = torch.cat([h.amax(dim=2), h[:, :, -RECENT:].mean(dim=2)], dim=1)
            return self.fc(pooled)

    class GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_in, HIDDEN, batch_first=True)
            self.fc = nn.Linear(HIDDEN, len(CLASSES))

        def forward(self, x):
            _, h = self.gru(x)
            return self.fc(h[0])

    return {"cnn": CNN, "gru": GRU}[kind]()


def train(kind, X, y, seed=0):
    import torch

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    F = inputs(X)
    mu, sd = F.reshape(-1, F.shape[2]).mean(0), F.reshape(-1, F.shape[2]).std(0) + 1e-6
    Fs = torch.tensor((F - mu) / sd)
    Y = torch.tensor([CLASSES.index(c) for c in y])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = torch_net(kind, F.shape[2])
    counts = np.bincount(Y.numpy(), minlength=len(CLASSES))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(Y) / (len(CLASSES) * counts), dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    net.train()
    for _ in range(EPOCHS):
        for b in np.array_split(rng.permutation(len(Y)), max(1, len(Y) // BATCH)):
            xb = Fs[b] + NOISE * torch.randn_like(Fs[b])
            opt.zero_grad()
            loss_fn(net(xb), Y[b]).backward()
            opt.step()
    net.eval()
    params = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    cls = {"cnn": CNNClassifier, "gru": GRUClassifier}[kind]
    model = cls(params, mu, sd)
    with torch.no_grad():
        ref = torch.softmax(net(Fs[:256]), dim=1).numpy()
    got = softmax(model.logits(Fs[:256].numpy()))
    assert np.abs(ref - got).max() < 1e-4, f"numpy {kind} forward disagrees with PyTorch: {np.abs(ref - got).max()}"
    return model


class _DeepClassifier(WindowClassifier):
    forward = None

    def __init__(self, params, mu, sd):
        super().__init__()
        self.params = {k: np.asarray(v, np.float32) for k, v in params.items()}
        self.mu, self.sd = np.asarray(mu, np.float32), np.asarray(sd, np.float32)

    def logits(self, Fs):
        return type(self).forward(Fs, self.params)

    def predict_windows(self, X):
        return softmax(self.logits((inputs(X) - self.mu) / self.sd).astype(np.float64))

    def to_json(self):
        return {"inputs": {"joints": JOINTS, "velocities": True, "recent": RECENT},
                "weights": {k: v.tolist() for k, v in self.params.items()},
                "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_json(cls, doc):
        if doc["inputs"] != {"joints": JOINTS, "velocities": True, "recent": RECENT}:
            raise SystemExit("model was trained on different inputs")
        return cls(doc["weights"], doc["mu"], doc["sd"])


class CNNClassifier(_DeepClassifier):
    type = "cnn"
    forward = staticmethod(cnn_forward)


class GRUClassifier(_DeepClassifier):
    type = "gru"
    forward = staticmethod(gru_forward)


def train_on(kind, sessions, seed=0, labels_dir="data/labels"):
    X, y, _ = load(sessions, labels_dir=labels_dir)
    Xm, ym = mirror(X, y)
    return train(kind, np.concatenate([X, Xm]), np.concatenate([y, ym]), seed=seed)


def main():
    p = argparse.ArgumentParser(description="Train and evaluate the CNN or GRU gesture classifier.")
    p.add_argument("kind", choices=["cnn", "gru"])
    p.add_argument("--train", nargs="*")
    p.add_argument("--test", nargs="*")
    args = p.parse_args()

    labelled = sorted(os.path.basename(f)[:-5] for f in glob.glob("data/labels/*.json"))
    splits = ([(args.train, s) for s in args.test] if args.train and args.test
              else [([t for t in labelled if t != s], s) for s in labelled])
    for train_s, test in splits:
        print(f"\n=== {args.kind}: train {'+'.join(train_s)} -> test {test} ===")
        evaluate(train_on(args.kind, train_s), test, PRESETS[args.kind]["balanced"]["trigger"])

    model = train_on(args.kind, args.train or labelled)
    path = f"models/{args.kind}.json"
    save_model(model, path, presets=PRESETS[args.kind])
    print(f"\nsaved {args.kind} trained on {'+'.join(args.train or labelled)} to {path}")


if __name__ == "__main__":
    main()
