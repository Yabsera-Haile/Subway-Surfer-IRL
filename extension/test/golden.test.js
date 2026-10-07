import { readdirSync, readFileSync } from "node:fs";
import { modelFromDoc, run } from "../pipeline.js";

const dir = new URL("../../tests/", import.meta.url);
const files = readdirSync(dir).filter(f => /^golden_.*\.json$/.test(f)).sort();
if (!files.length) {
  console.log("no golden files in tests/ - run gestures/test_pipeline.py --export");
  process.exit(1);
}

let allOk = true;
for (const file of files) {
  const golden = JSON.parse(readFileSync(new URL(file, dir), "utf8"));
  const { model, trigger } = modelFromDoc(golden.model);
  const frames = golden.input.map(x => [x.t, x.ok, x.ok ? Float64Array.from(x.landmarks.flat(), v => Math.fround(v)) : null]);
  const t0 = performance.now();
  const got = [...run(frames, model, trigger)];
  const msPerFrame = (performance.now() - t0) / got.length;
  const want = golden.expected;

  const problems = [];
  let worst = 0;
  if (got.length !== want.length) problems.push(`${got.length} output frames, expected ${want.length}`);
  for (let i = 0; i < Math.min(got.length, want.length); i++) {
    const g = got[i], w = want[i];
    if (Math.abs(g.t - w.t) > 1e-6) { problems.push(`frame ${i}: time ${g.t} vs ${w.t}`); break; }
    if ((g.probs === null) !== (w.probs === null)) problems.push(`t=${w.t}: probabilities present/absent mismatch`);
    else if (g.probs) g.probs.forEach((p, c) => { worst = Math.max(worst, Math.abs(p - w.probs[c])); });
    if (g.event !== w.event) problems.push(`t=${w.t}: event ${g.event} vs ${w.event}`);
  }

  const events = got.filter(g => g.event).map(g => `${g.event}@${g.t.toFixed(2)}`);
  const ok = problems.length === 0 && worst < golden.tolerance;
  allOk &&= ok;
  console.log(`${file}: ${got.length} frames, max |prob diff| ${worst.toExponential(2)}, ` +
    `${msPerFrame.toFixed(2)} ms/frame, events [${events.join(", ")}] -> ${ok ? "PASS" : "FAIL"}`);
  if (problems.length) console.log("  problems:", problems.slice(0, 5));
}
console.log(allOk ? "PASS" : "FAIL");
process.exit(allOk ? 0 : 1);
