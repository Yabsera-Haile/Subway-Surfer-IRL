function player(gl) {
  return {
    translate: [0.0, -0.70, -3.15],
    rotation: 0,
    type: "mono",
    score: 0,
    jumpboost: false,
    flyboost: false,
    jumpheight: -0.15,
    airT: null,
  };
}

function player_tick(object, obstacles) {
  if (statusKeys[37] || statusKeys[65]) {
    statusKeys[37] = false;
    statusKeys[65] = false;
    if (object.translate[0] == 0.0) object.translate[0] = -1.05;
    else if (object.translate[0] == 1.05) object.translate[0] = 0.0;
  }
  if (statusKeys[39] || statusKeys[68]) {
    statusKeys[39] = false;
    statusKeys[68] = false;
    if (object.translate[0] == 0.0) object.translate[0] = 1.05;
    else if (object.translate[0] == -1.05) object.translate[0] = 0.0;
  }

  if (statusKeys[32] || statusKeys[38] || statusKeys[87]) {
    statusKeys[32] = statusKeys[38] = statusKeys[87] = false;
    if (grounded(object)) object.airT = 0;
  }

  if (object.flyboost) {
    object.translate[1] = object.jumpheight;
    object.airT = CONFIG.jumpS / 2;
    return;
  }
  if (object.airT == null) return;

  const T = CONFIG.jumpS * (object.jumpboost ? 1.25 : 1);
  let t = object.airT + STEP;
  if (object.jumpboost && t > T / 2 && train_below(object, obstacles)) t = T / 2;
  if (t >= T) {
    object.airT = null;
    object.translate[1] = GROUND_Y;
  } else {
    object.airT = t;
    object.translate[1] = GROUND_Y + (object.jumpheight - GROUND_Y) * 4 * (t / T) * (1 - t / T);
  }
}

function train_below(player, obstacles) {
  return obstacles.some(o => o.translate[0] == player.translate[0] && zOverlap(player, o, 0.75));
}
