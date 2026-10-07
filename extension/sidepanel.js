import { PoseLandmarker } from "./vendor/mediapipe/vision_bundle.js";
import { CLASSES, GESTURES, MAX_INTERP_S, N_LANDMARKS, Resampler, modelFromDoc } from "./pipeline.js";

const SOURCE = "lane-gestures";
const W = 320, H = 240;
const ASPECT = 4 / 3;
const HOLD_MS = 100;
const VIS = 0.5;
const BENCH_FRAMES = 25;
const MIN_FPS = Math.ceil(1 / MAX_INTERP_S) + 1;
const CONNECTIONS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
                     [23, 24], [23, 25], [24, 26], [25, 27], [26, 28]];
const PAIRS = [[1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18],
               [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]];
const SWAP = Array.from({ length: N_LANDMARKS }, (_, i) => i);
for (const [a, b] of PAIRS) { SWAP[a] = b; SWAP[b] = a; }
const UP = { key: "ArrowUp", code: "ArrowUp", keyCode: 38 };
const DOWN = { key: "ArrowDown", code: "ArrowDown", keyCode: 40 };
const SPACE = { key: " ", code: "Space", keyCode: 32 };
const KEYMAPS = {
  arrows: {
    LEFT: { key: "ArrowLeft", code: "ArrowLeft", keyCode: 37 },
    RIGHT: { key: "ArrowRight", code: "ArrowRight", keyCode: 39 },
    JUMP: UP,
    DUCK: DOWN,
  },
  wasd: {
    LEFT: { key: "a", code: "KeyA", keyCode: 65 },
    RIGHT: { key: "d", code: "KeyD", keyCode: 68 },
    JUMP: { key: "w", code: "KeyW", keyCode: 87 },
    DUCK: { key: "s", code: "KeyS", keyCode: 83 },
  },
  "dino-up": { LEFT: null, RIGHT: null, JUMP: { ...UP, hold: 300 }, DUCK: { ...DOWN, hold: 600 } },
  "dino-space": { LEFT: null, RIGHT: null, JUMP: { ...SPACE, hold: 300 }, DUCK: { ...DOWN, hold: 600 } },
};

const HARMLESS = [
  "OpenGL error checking is disabled",
  "Created TensorFlow Lite XNNPACK delegate",
  "Feedback manager requires a model with a single signature",
  "Using NORM_RECT without IMAGE_DIMENSIONS",
];
globalThis.dbg = (...args) => {
  const text = args.join(" ");
  if (!HARMLESS.some(h => text.includes(h))) console.warn(...args);
};

const $ = id => document.getElementById(id);
const view = $("view"), ctx = view.getContext("2d"), video = $("video");

const engines = {};
let landmarker = null, engine = null, model = null, trigger = null;
const resampler = new Resampler();
let stream = null, running = false, lastTs = -1;
let target = null;
let settings = { keymap: "arrows", send: true, engine: "auto", model: "baseline", speed: "balanced" };
const fps = { frames: 0, since: performance.now(), value: 0, cameraMs: 0, poseMs: 0, workMs: 0 };

function setStatus(el, text, kind = "") {
  el.textContent = text;
  el.className = `status ${kind}`;
}

function buildBars() {
  $("bars").innerHTML = CLASSES.map(c => `
    <span>${c}</span>
    <div class="track"><div class="fill${c === "IDLE" ? " idle" : ""}" id="bar-${c}"></div></div>
    <span class="num" id="num-${c}">–</span>`).join("");
}

function showProbs(probs) {
  CLASSES.forEach((c, i) => {
    const p = probs ? probs[i] : 0;
    $(`bar-${c}`).style.width = `${(p * 100).toFixed(0)}%`;
    $(`num-${c}`).textContent = probs ? p.toFixed(2) : "–";
  });
}

let flashTimer = null;
function flash(text) {
  $("flash").textContent = text;
  $("flash").hidden = false;
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { $("flash").hidden = true; }, 600);
}

const MODEL_NAMES = { baseline: "Baseline", cnn: "CNN", gru: "GRU" };
const SPEED_NAMES = { balanced: "Balanced", fast: "Fast", fastest: "Fastest" };
const docs = {};

// Loads the selected model file and shows its measured accuracy and latency.
async function loadModel() {
  const name = settings.model;
  docs[name] ??= await (await fetch(chrome.runtime.getURL(`models/${name}.json`))).json();
  ({ model, trigger } = modelFromDoc(docs[name], settings.speed));
  const m = docs[name].presets?.[settings.speed]?.measured;
  setStatus($("modelinfo"), m
    ? `${MODEL_NAMES[name]} · ${SPEED_NAMES[settings.speed]}: fires ~${m.latency_s.toFixed(2)} s after you move, ` +
      `caught ${(m.caught * 100).toFixed(0)}%, ${m.false_per_min.toFixed(1)} false moves/min ` +
      `(trained on one session, tested on the other).`
    : `${MODEL_NAMES[name]}: no measured settings in this model file.`);
}

async function loadEngine(d) {
  if (engines[d]) return engines[d];
  const fileset = {
    wasmLoaderPath: chrome.runtime.getURL("vendor/mediapipe/wasm/vision_wasm_internal.js"),
    wasmBinaryPath: chrome.runtime.getURL("vendor/mediapipe/wasm/vision_wasm_internal.wasm"),
  };
  engines[d] = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: chrome.runtime.getURL("models/pose_landmarker_lite.task"), delegate: d },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputSegmentationMasks: false,
  });
  return engines[d];
}

function nextFrame() {
  return new Promise(resolve => video.requestVideoFrameCallback((now, meta) => resolve({ now, meta })));
}

function nextTimestamp(t) {
  lastTs = Math.max(Math.round(t), lastTs + 1);
  return lastTs;
}

async function timeEngine(lm) {
  const ms = [];
  for (let i = 0; i < BENCH_FRAMES; i++) {
    const { now } = await nextFrame();
    const t0 = performance.now();
    lm.detectForVideo(video, nextTimestamp(now));
    ms.push(performance.now() - t0);
  }
  ms.splice(0, 5);
  ms.sort((a, b) => a - b);
  return ms[ms.length >> 1];
}

// Loads the GPU and/or CPU pose engine and keeps the faster one.
async function chooseEngine() {
  const wanted = settings.engine === "auto" ? ["GPU", "CPU"] : [settings.engine];
  const timings = {};
  for (const d of wanted) {
    try {
      setStatus($("status"), `Loading pose model (${d})…`);
      const lm = await loadEngine(d);
      if (wanted.length > 1) {
        setStatus($("status"), `Measuring ${d} speed - stand in view…`);
        timings[d] = await timeEngine(lm);
      }
    } catch (e) {
      console.warn(`${d} pose engine unavailable:`, e);
    }
  }
  const usable = wanted.filter(d => engines[d]);
  if (!usable.length) throw new Error("no pose engine could start");
  engine = usable.reduce((best, d) => ((timings[d] ?? Infinity) < (timings[best] ?? Infinity) ? d : best));
  landmarker = engines[engine];
  for (const d of Object.keys(engines)) {
    if (d !== engine) { engines[d].close(); delete engines[d]; }
  }
  return Object.entries(timings).map(([d, ms]) => `${d} ${ms.toFixed(0)} ms`).join(", ");
}

async function start() {
  try {
    setStatus($("status"), "Starting camera…");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false,
    });
  } catch (e) {
    if (e.name === "NotAllowedError") {
      setStatus($("status"), "The extension needs camera access. Grant it once, then press Start again.", "error");
      $("grant").hidden = false;
    } else if (e.name === "NotFoundError") {
      setStatus($("status"), "No camera found. Plug one in and press Start again.", "error");
    } else {
      setStatus($("status"), `Could not start: ${e.message}`, "error");
    }
    return;
  }
  $("grant").hidden = true;
  video.srcObject = stream;
  await video.play();
  $("start").textContent = "Stop camera";
  running = true;

  let picked = "";
  try {
    picked = await chooseEngine();
  } catch (e) {
    stop();
    setStatus($("status"), `Could not start pose detection: ${e.message}`, "error");
    return;
  }
  if (!running) return;
  if (picked) setStatus($("status"), `Using ${engine} (measured ${picked}).`, "ok");
  resampler.reset();
  model.reset();
  trigger.reset();
  findGame();
  video.requestVideoFrameCallback(onFrame);
}

function stop() {
  running = false;
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
  showProbs(null);
  $("start").textContent = "Start camera";
  setStatus($("status"), "Camera stopped.");
}

function cropBox() {
  const vw = video.videoWidth, vh = video.videoHeight;
  return vw / vh > ASPECT
    ? { sx: (vw - vh * ASPECT) / 2, sy: 0, sw: vh * ASPECT, sh: vh, vw, vh }
    : { sx: 0, sy: (vh - vw / ASPECT) / 2, sw: vw, sh: vw / ASPECT, vw, vh };
}

// Crops to 4:3, mirrors and swaps left/right landmarks to match the training data.
function toTrainingFrame(pose, box) {
  const out = new Float64Array(N_LANDMARKS * 4);
  for (let j = 0; j < N_LANDMARKS; j++) {
    const p = pose[SWAP[j]];
    out[j * 4] = 1 - (p.x * box.vw - box.sx) / box.sw;
    out[j * 4 + 1] = (p.y * box.vh - box.sy) / box.sh;
    out[j * 4 + 2] = p.z;
    out[j * 4 + 3] = p.visibility;
  }
  return out;
}

function drawPreview(box, lm) {
  ctx.setTransform(-1, 0, 0, 1, W, 0);
  ctx.drawImage(video, box.sx, box.sy, box.sw, box.sh, 0, 0, W, H);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (!lm) return;
  const x = j => lm[j * 4] * W, y = j => lm[j * 4 + 1] * H, seen = j => lm[j * 4 + 3] >= VIS;
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ffd23f";
  for (const [a, b] of CONNECTIONS) {
    if (!seen(a) || !seen(b)) continue;
    ctx.beginPath();
    ctx.moveTo(x(a), y(a));
    ctx.lineTo(x(b), y(b));
    ctx.stroke();
  }
  ctx.fillStyle = "#ff5a36";
  for (const j of [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
    if (!seen(j)) continue;
    ctx.beginPath();
    ctx.arc(x(j), y(j), 3, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function onFrame(now, meta) {
  if (!running) return;
  const start = performance.now();
  const cap = meta?.captureTime;
  const captured = cap && cap <= now && now - cap < 2000 ? cap : now;

  const box = cropBox();
  const pose = landmarker.detectForVideo(video, nextTimestamp(captured)).landmarks[0];
  const poseMs = performance.now() - start;
  const lm = pose ? toTrainingFrame(pose, box) : null;

  let probs;
  for (const f of resampler.push(captured / 1000, !!pose, lm)) {
    probs = model.step(f.frame, f.valid);
    const event = trigger.step(f.t, probs);
    if (event) onGesture(event, captured);
  }
  if (probs !== undefined) showProbs(probs);
  drawPreview(box, lm);

  const smooth = (old, v) => (old ? 0.9 * old + 0.1 * v : v);
  fps.cameraMs = smooth(fps.cameraMs, start - captured);
  fps.poseMs = smooth(fps.poseMs, poseMs);
  fps.workMs = smooth(fps.workMs, performance.now() - start);
  fps.frames++;
  if (now - fps.since > 1000) {
    fps.value = (fps.frames * 1000) / (now - fps.since);
    fps.frames = 0;
    fps.since = now;
    const camera = cap ? ` · camera ${fps.cameraMs.toFixed(0)} ms` : "";
    const where = pose ? "tracking you" : "no one in view";
    const numbers = `${fps.value.toFixed(0)} fps · pose ${fps.poseMs.toFixed(0)} ms (${engine}) · total ${fps.workMs.toFixed(0)} ms${camera}`;
    if (fps.value < MIN_FPS) {
      const fix = engine === "GPU"
        ? "The pose model shares the GPU with the game: set the game's graphics to Low, or pick Pose engine CPU and restart the camera."
        : "Close other heavy tabs or apps, or pick Pose engine GPU and restart the camera.";
      setStatus($("status"), `Too slow for gestures (needs ${MIN_FPS}+ fps): ${numbers}. ${fix}`, "error");
    } else {
      setStatus($("status"), `${numbers} · ${where}`, fps.value < 20 ? "error" : "ok");
    }
  }
  video.requestVideoFrameCallback(onFrame);
}

async function findGame() {
  target = null;
  setStatus($("target"), "Looking for a game canvas…");
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return setStatus($("target"), "No active tab.", "error");
  let frames = [];
  try {
    frames = await chrome.webNavigation.getAllFrames({ tabId: tab.id });
  } catch {
    return setStatus($("target"), "This tab can't be controlled (browser pages and the Web Store are off limits).", "error");
  }
  const found = await Promise.all(frames.map(async f => {
    try {
      const info = await chrome.tabs.sendMessage(tab.id, { source: SOURCE, type: "probe" }, { frameId: f.frameId });
      return info && { tabId: tab.id, frameId: f.frameId, info };
    } catch {
      return null;
    }
  }));
  const best = found.filter(Boolean).sort((a, b) => b.info.width * b.info.height - a.info.width * a.info.height)[0];
  if (!best) {
    return setStatus($("target"), "No game canvas found. Start the game, or reload the tab if it was open before installing.", "error");
  }
  target = best;
  setStatus($("target"), `Keys go to ${best.info.host} (canvas ${best.info.width}×${best.info.height}).`, "ok");
}

async function sendKey(gesture) {
  const key = (KEYMAPS[settings.keymap] ?? KEYMAPS.arrows)[gesture];
  if (!key) return false;
  if (!target) await findGame();
  if (!target) return false;
  try {
    const info = await chrome.tabs.sendMessage(target.tabId,
      { source: SOURCE, type: "key", key, holdMs: key.hold ?? HOLD_MS },
      { frameId: target.frameId });
    if (!info) target = null;
    return !!info;
  } catch {
    target = null;
    return false;
  }
}

async function onGesture(gesture, captured) {
  flash(gesture);
  if (!settings.send) return;
  if (await sendKey(gesture)) {
    const ms = performance.now() - captured;
    setStatus($("target"), `${gesture} reached the game ${ms.toFixed(0)} ms after the camera saw it ` +
      `(${target.info.host}).`, "ok");
  }
}

async function testKeys() {
  const map = KEYMAPS[settings.keymap] ?? KEYMAPS.arrows;
  const used = GESTURES.filter(g => map[g]);
  const label = g => (map[g].hold ? `${g} (held ${map[g].hold} ms)` : g);
  setStatus($("target"), `Click inside the game now - testing ${used.map(label).join(", ")} in 3 s…`);
  await new Promise(r => setTimeout(r, 3000));
  const sent = [];
  for (const g of used) {
    flash(g);
    if (await sendKey(g)) sent.push(label(g));
    await new Promise(r => setTimeout(r, 1200 + (map[g].hold ?? HOLD_MS)));
  }
  if (sent.length) setStatus($("target"), `Sent ${sent.join(", ")} to ${target.info.host}.`, "ok");
}

async function init() {
  buildBars();
  const saved = await chrome.storage.local.get(["keymap", "send", "engine", "model", "speed"]);
  settings = { ...settings, ...saved };
  $("keymap").value = settings.keymap;
  $("send").checked = settings.send;
  $("engine").value = settings.engine;
  $("model").value = settings.model;
  $("speed").value = settings.speed;
  for (const key of ["model", "speed"]) {
    $(key).onchange = async () => {
      const before = settings[key];
      settings[key] = $(key).value;
      try {
        await loadModel();
        chrome.storage.local.set({ [key]: settings[key] });
      } catch (e) {
        settings[key] = before;
        $(key).value = before;
        setStatus($("modelinfo"), `Could not load that model: ${e.message}`, "error");
      }
    };
  }
  $("keymap").onchange = () => { settings.keymap = $("keymap").value; chrome.storage.local.set({ keymap: settings.keymap }); };
  $("send").onchange = () => { settings.send = $("send").checked; chrome.storage.local.set({ send: settings.send }); };
  $("engine").onchange = () => {
    settings.engine = $("engine").value;
    chrome.storage.local.set({ engine: settings.engine });
    if (running) setStatus($("status"), "Pose engine changes when you next press Start.");
  };
  $("start").onclick = () => (running ? stop() : start());
  $("find").onclick = findGame;
  $("test").onclick = testKeys;
  $("grant").onclick = () => chrome.tabs.create({ url: chrome.runtime.getURL("permission.html") });
  chrome.tabs.onActivated.addListener(() => { target = null; if (running) findGame(); });
  chrome.tabs.onUpdated.addListener((_id, change) => { if (change.status === "complete") target = null; });

  try {
    try {
      await loadModel();
    } catch {
      settings.model = "baseline";
      $("model").value = "baseline";
      await loadModel();
    }
    setStatus($("status"), "Ready. Press Start camera.");
  } catch (e) {
    setStatus($("status"), `Could not load the gesture model: ${e.message}`, "error");
    $("start").disabled = true;
  }
}

init();
