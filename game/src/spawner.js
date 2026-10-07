const SPAWN_Z = -40;

var traveled = 0;
var nextRowAt = 0;
var prevHazardLanes = [];

function spawner_reset() {
  traveled = 0;
  nextRowAt = 3 * CONFIG.speed;
  prevHazardLanes = [];
}

function shuffled(list) {
  const a = list.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = getRandomInt(0, i);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function spawner_tick(gl) {
  traveled += speed;
  if (traveled < nextRowAt) return;
  const gapS = CONFIG.gapS[0] + Math.random() * (CONFIG.gapS[1] - CONFIG.gapS[0]);
  const gapDist = gapS * speed / STEP;
  spawn_row(gl, gapDist);
  nextRowAt = traveled + gapDist;
}

function spawn_row(gl, gapDist) {
  const lanes = shuffled(LANES);
  const n = CONFIG.hazardsPerRow > 1 && Math.random() < 0.3 ? 2 : 1;
  const hazardLanes = lanes.slice(0, n);
  for (const lane of hazardLanes) {
    if (Math.random() < CONFIG.trainShare) {
      obstacles.push(obstacle(gl, lane, SPAWN_Z));
    } else {
      barriers.push(barrier(gl, lane, SPAWN_Z));
    }
  }

  const safe = lanes.slice(n).filter(l => !prevHazardLanes.includes(l));
  if (safe.length) {
    const lane = safe[getRandomInt(0, safe.length - 1)];
    for (let z = SPAWN_Z + 1.5; z < SPAWN_Z + gapDist - 1.5; z += 2) {
      coins.push(coin(gl, lane, z));
    }
    if (boosts.length < 2 && Math.random() < 0.08) {
      boosts.push(boost(gl, lane, SPAWN_Z + gapDist / 2));
    }
  }
  prevHazardLanes = hazardLanes;
}
