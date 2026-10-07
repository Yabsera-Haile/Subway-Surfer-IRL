const fs = require("fs"), vm = require("vm"), path = require("path");
const SRC = path.join(__dirname, "..", "src");
const FILES = ["utility", "config", "rules", "spawner", "finishline", "boost", "obstacle",
               "barrier", "coin", "police", "player"];

function makeGame(difficulty, seedRandom) {
  const timers = [];
  const ctx = {
    console, Math: Object.create(Math),
    setTimeout: (fn, ms) => timers.push({ at: ctx.simTime + ms / 1000, fn }),
  };
  ctx.Math.random = seedRandom;
  vm.createContext(ctx);
  for (const f of FILES) vm.runInContext(fs.readFileSync(path.join(SRC, f + ".js"), "utf8"), ctx, { filename: f + ".js" });
  vm.runInContext(`
    var statusKeys = {}, game_over = false, game_start = true, finish = 0, speed = 0.075;
    var objects = [], coins = [], obstacles = [], barriers = [], boosts = [];
    var gl = null;
    setDifficulty(${JSON.stringify(difficulty)});
  `, ctx);
  vm.runInContext(`
    objects.push(player(gl)); objects.push(police(gl));
    finish_object = finishline(gl);
    spawner_reset();
  `, ctx);
  ctx.runTimers = () => {
    for (let i = timers.length - 1; i >= 0; i--) if (timers[i].at <= ctx.simTime) { const t = timers.splice(i, 1)[0]; t.fn(); }
  };
  return ctx;
}

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

// Plays one run with a bot whose key presses reach the game `latency` seconds late.
function play(difficulty, latency, seed, maxS = 180) {
  const g = makeGame(difficulty, rng(seed));
  const r = rng(seed * 7919 + 1);
  const P = () => g.objects[0];
  const queue = [];
  let lastInput = -9;
  const planned = new Set();
  let plannedLane = 0.0;
  const tti = (o, half) => ((P().translate[2] - 0.15) - (o.translate[2] + half)) / (g.speed * 60);
  const press = key => {
    const at = Math.max(g.simTime, lastInput + 0.5);
    lastInput = at;
    queue.push({ at: at + latency, key });
  };
  const trainIn = (lane, from, to) => g.obstacles.some(o => o.translate[0] == lane && tti(o, 0.75) > from && tti(o, 0.75) < to);

  while (g.simTime < maxS && !g.game_over) {
    for (let i = queue.length - 1; i >= 0; i--) if (queue[i].at <= g.simTime) { g.statusKeys[queue.splice(i, 1)[0].key] = true; }
    for (const [list, half, kind] of [[g.obstacles, 0.75, "train"], [g.barriers, 0.05, "barrier"]]) {
      for (const o of list) {
        if (planned.has(o) || o.translate[0] != plannedLane) continue;
        if (o.lead == null) o.lead = 0.4 + 0.4 * r();
        const t = tti(o, half);
        if (t > o.lead || t < -0.2) continue;
        planned.add(o);
        if (kind === "barrier") { press(32); continue; }
        const options = [-1.05, 0.0, 1.05].filter(l => l != plannedLane && !trainIn(l, -0.8, 1.6))
          .sort((a, b) => Math.abs(a - plannedLane) - Math.abs(b - plannedLane));
        const target = options[0];
        if (target == null) { press(32); continue; }
        const steps = Math.round((target - plannedLane) / 1.05);
        for (let k = 0; k < Math.abs(steps); k++) press(steps < 0 ? 37 : 39);
        plannedLane = target;
      }
    }
    g.step(g.gl);
    g.runTimers();
  }
  return { survived: g.simTime >= maxS, time: g.simTime, saves: g.saves, score: g.objects[0].score,
           caught: g.game_over && g.objects[1].setback, trains: g.game_over && !g.objects[1].setback };
}

const RUNS = 20;
console.log(`each cell: ${RUNS} runs of up to 3 min; a person reacting 0.4-0.8 s before each hazard, moves limited to one per 0.5 s\n`);
console.log("difficulty  gesture delay   survived 3 min   median time   ended by train / police   late dodges forgiven per run");
for (const d of ["gesture", "normal", "classic"]) {
  for (const latency of [0, 0.15, 0.30, 0.45]) {
    const res = Array.from({ length: RUNS }, (_, i) => play(d, latency, 1000 + i));
    const times = res.map(x => x.time).sort((a, b) => a - b);
    const med = times[Math.floor(times.length / 2)];
    const ok = res.filter(x => x.survived).length;
    const tr = res.filter(x => x.trains).length, po = res.filter(x => x.caught).length;
    const sv = res.reduce((a, x) => a + x.saves, 0) / RUNS;
    console.log(`${d.padEnd(11)} ${(latency.toFixed(2) + " s").padStart(13)}   ${(ok + "/" + RUNS).padStart(14)}   ${(med.toFixed(0) + " s").padStart(11)}   ${(tr + " / " + po).padStart(23)}   ${sv.toFixed(1).padStart(10)}`);
  }
}
