export const CLASSES = ["LEFT", "RIGHT", "JUMP", "DUCK", "IDLE"];
export const GESTURES = CLASSES.slice(0, 4);
export const HZ = 30;
export const WINDOW_N = 30;
export const MAX_INTERP_S = 0.10;
export const N_LANDMARKS = 33;

const NOSE = 0, L_SHOULDER = 11, R_SHOULDER = 12, L_WRIST = 15, R_WRIST = 16;
const L_HIP = 23, R_HIP = 24, L_KNEE = 25, R_KNEE = 26;

const RECENT = 9;
const SIGNALS = ["body x", "body y", "arms x", "arms y", "head y", "knee bend", "torso length"];
const STATS = ["end", "min", "max", "min velocity", "max velocity",
               "recent change", "recent min velocity", "recent max velocity"];
export const FEATURE_NAMES = SIGNALS.flatMap(s => STATS.map(t => `${s} ${t}`));

// Resamples raw pose frames onto the 30 Hz grid, marking frames across long gaps invalid.
export class Resampler {
  constructor() { this.reset(); }

  reset() { this.t = null; this.lm = null; this.k = null; }

  push(t, ok, landmarks) {
    if (!ok) return [];
    const out = [];
    if (this.t === null) {
      this.k = Math.ceil(t * HZ);
    } else {
      const valid = t - this.t <= MAX_INTERP_S;
      while (this.k / HZ <= t) {
        const g = this.k / HZ;
        const frame = new Float32Array(landmarks.length);
        for (let i = 0; i < frame.length; i++) {
          frame[i] = this.lm[i] + ((g - this.t) * (landmarks[i] - this.lm[i])) / (t - this.t);
        }
        out.push({ t: g, frame, valid });
        this.k++;
      }
    }
    this.t = t;
    this.lm = landmarks;
    return out;
  }
}

function median(values) {
  const s = Array.from(values).sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Makes a window's x,y relative to its starting body centre, in torso lengths.
export function normalizeWindow(win) {
  const T = win.length;
  const cx = new Float64Array(T), cy = new Float64Array(T), torso = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    const f = win[t];
    const sx = (f[L_SHOULDER * 4] + f[R_SHOULDER * 4]) / 2, sy = (f[L_SHOULDER * 4 + 1] + f[R_SHOULDER * 4 + 1]) / 2;
    const hx = (f[L_HIP * 4] + f[R_HIP * 4]) / 2, hy = (f[L_HIP * 4 + 1] + f[R_HIP * 4 + 1]) / 2;
    cx[t] = (sx + hx) / 2;
    cy[t] = (sy + hy) / 2;
    torso[t] = Math.hypot(sx - hx, sy - hy);
  }
  const ox = (cx[0] + cx[1] + cx[2]) / 3, oy = (cy[0] + cy[1] + cy[2]) / 3;
  const scale = median(torso);
  const Z = new Float64Array(T * N_LANDMARKS * 2);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < N_LANDMARKS; j++) {
      Z[(t * N_LANDMARKS + j) * 2] = (win[t][j * 4] - ox) / scale;
      Z[(t * N_LANDMARKS + j) * 2 + 1] = (win[t][j * 4 + 1] - oy) / scale;
    }
  }
  return Z;
}

export function baselineFeatures(Z, T = WINDOW_N) {
  const x = (t, j) => Z[(t * N_LANDMARKS + j) * 2];
  const y = (t, j) => Z[(t * N_LANDMARKS + j) * 2 + 1];
  const signals = SIGNALS.map(() => new Float64Array(T));
  for (let t = 0; t < T; t++) {
    const sx = (x(t, L_SHOULDER) + x(t, R_SHOULDER)) / 2, sy = (y(t, L_SHOULDER) + y(t, R_SHOULDER)) / 2;
    const hx = (x(t, L_HIP) + x(t, R_HIP)) / 2, hy = (y(t, L_HIP) + y(t, R_HIP)) / 2;
    const cx = (sx + hx) / 2, cy = (sy + hy) / 2;
    signals[0][t] = cx;
    signals[1][t] = cy;
    signals[2][t] = (x(t, L_WRIST) + x(t, R_WRIST)) / 2 - cx;
    signals[3][t] = (y(t, L_WRIST) + y(t, R_WRIST)) / 2 - cy;
    signals[4][t] = y(t, NOSE) - cy;
    signals[5][t] = (y(t, L_KNEE) + y(t, R_KNEE)) / 2 - hy;
    signals[6][t] = Math.hypot(sx - hx, sy - hy);
  }
  const out = [];
  for (const raw of signals) {
    const s = raw.map(v => v - raw[0]);
    let lo = Infinity, hi = -Infinity, vlo = Infinity, vhi = -Infinity, rlo = Infinity, rhi = -Infinity;
    for (let t = 0; t < T; t++) {
      lo = Math.min(lo, s[t]);
      hi = Math.max(hi, s[t]);
      if (t > 0) {
        const v = (s[t] - s[t - 1]) * HZ;
        vlo = Math.min(vlo, v);
        vhi = Math.max(vhi, v);
        if (t >= T - RECENT) {
          rlo = Math.min(rlo, v);
          rhi = Math.max(rhi, v);
        }
      }
    }
    out.push(s[T - 1], lo, hi, vlo, vhi, s[T - 1] - s[T - 1 - RECENT], rlo, rhi);
  }
  return out;
}

function softmax(z) {
  const m = Math.max(...z);
  const e = z.map(v => Math.exp(v - m));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / sum);
}

// Base class for models that classify the last WINDOW_N frames.
export class WindowClassifier {
  constructor() { this.reset(); }

  reset() { this.buf = []; }

  step(frame, valid) {
    if (!valid) {
      this.buf = [];
      return null;
    }
    this.buf.push(frame);
    if (this.buf.length > WINDOW_N) this.buf.shift();
    return this.buf.length < WINDOW_N ? null : this.predictWindow(this.buf);
  }
}

export class BaselineClassifier extends WindowClassifier {
  static fromDoc(doc) {
    if (JSON.stringify(doc.features) !== JSON.stringify(FEATURE_NAMES)) {
      throw new Error("model was trained on a different feature set than this extension computes");
    }
    return new BaselineClassifier(doc.weights);
  }

  constructor({ W, b, mu, sd }) {
    super();
    Object.assign(this, { W, b, mu, sd });
  }

  predictWindow(win) {
    const F = baselineFeatures(normalizeWindow(win), win.length);
    const z = this.b.slice();
    for (let i = 0; i < F.length; i++) {
      const f = (F[i] - this.mu[i]) / this.sd[i];
      for (let c = 0; c < z.length; c++) z[c] += f * this.W[i][c];
    }
    return softmax(z);
  }
}

// Turns a stream of class probabilities into one event per gesture.
export class Trigger {
  constructor({ thresh, hold, release, cooldown_s }) {
    Object.assign(this, { thresh, hold, release, cooldownS: cooldown_s });
    this.reset();
  }

  reset() {
    this.run = new Array(GESTURES.length).fill(0);
    this.last = -Infinity;
    this.armed = true;
  }

  step(t, probs) {
    if (!probs) {
      this.run.fill(0);
      return null;
    }
    const g = probs.slice(0, GESTURES.length);
    this.armed = this.armed || Math.max(...g) < this.release;
    this.run = g.map((p, i) => (p > this.thresh ? this.run[i] + 1 : 0));
    if (!this.armed || t - this.last < this.cooldownS) return null;
    let best = -1;
    for (let i = 0; i < g.length; i++) {
      if (this.run[i] >= this.hold && (best < 0 || g[i] > g[best])) best = i;
    }
    if (best < 0) return null;
    this.last = t;
    this.armed = false;
    this.run.fill(0);
    return GESTURES[best];
  }
}

const DEEP_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const DEEP_INPUTS = JSON.stringify({ joints: DEEP_JOINTS, velocities: true, recent: RECENT });

// Builds the neural models' input: 13 joints' positions and velocities, standardized.
function deepInputs(win, mu, sd) {
  const Z = normalizeWindow(win);
  const n = DEEP_JOINTS.length * 2;
  const rows = [];
  for (let t = 0; t < win.length; t++) {
    const row = new Float64Array(2 * n);
    DEEP_JOINTS.forEach((j, i) => {
      row[2 * i] = Z[(t * N_LANDMARKS + j) * 2];
      row[2 * i + 1] = Z[(t * N_LANDMARKS + j) * 2 + 1];
    });
    for (let c = 0; c < n; c++) row[n + c] = t ? row[c] - rows[t - 1][c] : 0;
    rows.push(row);
  }
  return rows.map(row => row.map((v, c) => (v - mu[c]) / sd[c]));
}

// 1D convolution with zero 'same' padding; W is laid out [out][in][k] like PyTorch.
function conv1d(x, W, b) {
  const T = x.length, K = W[0][0].length, pad = K >> 1;
  return x.map((_, t) => Float64Array.from(W, (wo, o) => {
    let s = b[o];
    for (let k = 0; k < K; k++) {
      const src = t + k - pad;
      if (src < 0 || src >= T) continue;
      const xs = x[src];
      for (let c = 0; c < xs.length; c++) s += wo[c][k] * xs[c];
    }
    return s;
  }));
}

function linear(x, W, b) {
  return W.map((row, o) => row.reduce((s, w, i) => s + w * x[i], b[o]));
}

const sigmoid = v => 1 / (1 + Math.exp(-v));

class DeepClassifier extends WindowClassifier {
  static fromDoc(doc) {
    if (JSON.stringify(doc.inputs) !== DEEP_INPUTS) {
      throw new Error("model was trained on different inputs than this extension computes");
    }
    return new this(doc.weights, doc.mu, doc.sd);
  }

  constructor(weights, mu, sd) {
    super();
    Object.assign(this, { p: weights, mu, sd });
  }

  predictWindow(win) {
    return softmax(this.forward(deepInputs(win, this.mu, this.sd)));
  }
}

class CNNClassifier extends DeepClassifier {
  forward(F) {
    const p = this.p;
    const relu = rows => rows.map(r => r.map(v => Math.max(v, 0)));
    const h = relu(conv1d(relu(conv1d(F, p["conv1.weight"], p["conv1.bias"])), p["conv2.weight"], p["conv2.bias"]));
    const C = h[0].length;
    const pooled = new Float64Array(2 * C);
    for (let c = 0; c < C; c++) {
      let peak = -Infinity, recent = 0;
      for (let t = 0; t < h.length; t++) peak = Math.max(peak, h[t][c]);
      for (let t = h.length - RECENT; t < h.length; t++) recent += h[t][c];
      pooled[c] = peak;
      pooled[C + c] = recent / RECENT;
    }
    return linear(pooled, p["fc.weight"], p["fc.bias"]);
  }
}

class GRUClassifier extends DeepClassifier {
  forward(F) {
    const p = this.p;
    const Wi = p["gru.weight_ih_l0"], Wh = p["gru.weight_hh_l0"], bi = p["gru.bias_ih_l0"], bh = p["gru.bias_hh_l0"];
    const H = Wh[0].length;
    let h = new Float64Array(H);
    for (const x of F) {
      const gi = linear(x, Wi, bi), gh = linear(h, Wh, bh);
      const next = new Float64Array(H);
      for (let k = 0; k < H; k++) {
        const r = sigmoid(gi[k] + gh[k]);
        const z = sigmoid(gi[H + k] + gh[H + k]);
        const n = Math.tanh(gi[2 * H + k] + r * gh[2 * H + k]);
        next[k] = (1 - z) * n + z * h[k];
      }
      h = next;
    }
    return linear(h, p["fc.weight"], p["fc.bias"]);
  }
}

const MODEL_TYPES = { baseline: BaselineClassifier, cnn: CNNClassifier, gru: GRUClassifier };

// Builds the classifier and trigger described by a model file and preset name.
export function modelFromDoc(doc, preset) {
  for (const [key, want] of [["classes", CLASSES], ["hz", HZ], ["window_n", WINDOW_N]]) {
    if (JSON.stringify(doc[key]) !== JSON.stringify(want)) {
      throw new Error(`model was made with ${key}=${JSON.stringify(doc[key])}, this extension uses ${JSON.stringify(want)}`);
    }
  }
  const Model = MODEL_TYPES[doc.type];
  if (!Model) throw new Error(`unknown model type "${doc.type}"`);
  const params = doc.presets?.[preset]?.trigger ?? doc.trigger;
  return { model: Model.fromDoc(doc), trigger: new Trigger(params) };
}

// Runs [t, ok, landmarks] frames through the model and trigger, yielding { t, probs, event }.
export function* run(frames, model, trigger) {
  const resampler = new Resampler();
  model.reset();
  trigger.reset();
  for (const [t, ok, lm] of frames) {
    for (const f of resampler.push(t, ok, lm)) {
      const probs = model.step(f.frame, f.valid);
      yield { t: f.t, probs, event: trigger.step(f.t, probs) };
    }
  }
}
