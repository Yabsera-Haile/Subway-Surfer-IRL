const STEP = 1 / 60;
const GROUND_Y = -0.70;
const LANES = [-1.05, 0.0, 1.05];

var simTime = 0;
var chaseUntil = -1;
var slowUntil = -1;
var saves = 0;

function grounded(p) {
  return p.airT == null && !p.flyboost && p.translate[1] <= GROUND_Y + 1e-9;
}

function zOverlap(p, o, halfDepth) {
  return !(p.translate[2] - 0.15 >= o.translate[2] + halfDepth || p.translate[2] + 0.15 <= o.translate[2] - halfDepth);
}

function currentSpeed() {
  const perSec = Math.min(CONFIG.maxSpeed, CONFIG.speed + CONFIG.rampPerMin * simTime / 60);
  return perSec * (simTime < slowUntil ? CONFIG.stumbleSlow : 1) * STEP;
}

function stumble(police) {
  if (CONFIG.chase === "forgiving" && simTime < chaseUntil) {
    game_over = true;
    return;
  }
  chaseUntil = simTime + CONFIG.chaseS;
  slowUntil = simTime + CONFIG.stumbleSlowS;
  police.setback = true;
}

function step(gl) {
  simTime += STEP;
  speed = currentSpeed();
  if (objects[1].setback && simTime > chaseUntil) objects[1].setback = false;

  spawner_tick(gl);
  obstacle_tick(gl, obstacles, objects[0]);
  barrier_tick(gl, barriers, objects[0], objects[1]);
  coin_tick(gl, coins, objects[0]);
  player_tick(objects[0], obstacles);
  police_tick(objects[1], objects[0]);
  boost_tick(gl, boosts, objects[0]);
  if (objects[0].score >= 100 && !finish && !game_over) finishline_tick(finish_object, objects[0]);
}
