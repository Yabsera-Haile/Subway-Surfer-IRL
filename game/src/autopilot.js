// Bot player for ?autopilot=1: changes lane before trains and jumps barriers.
function autopilot_tick() {
  const p = objects[0];
  const ups = speed / STEP || 1;
  const until = (o, half) => ((p.translate[2] - 0.15) - (o.translate[2] + half)) / ups;
  const trainIn = (lane, from, to) => obstacles.some(o => o.translate[0] === lane &&
    until(o, 0.75) > from && until(o, 0.75) < to);

  if (trainIn(p.translate[0], -0.3, 0.7)) {
    const way = LANES.filter(l => l !== p.translate[0] && Math.abs(l - p.translate[0]) < 1.1 && !trainIn(l, -0.6, 1.2));
    if (way.length) statusKeys[way[0] < p.translate[0] ? 37 : 39] = true;
  }
  if (barriers.some(b => b.translate[0] === p.translate[0] && until(b, 0.05) > 0 && until(b, 0.05) < 0.25)) {
    statusKeys[32] = true;
  }
}
