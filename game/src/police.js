function police(gl) {
  return { translate: [0.0, -0.70, -2.15], type: "police", setback: false };
}

function police_tick(object, player) {
  object.translate[0] = player.translate[0];

  if (CONFIG.chase === "classic") {
    if (object.setback == false) {
      object.translate[2] += 0.005;
    }
    if (object.setback == true) {
      object.translate[2] -= 0.005;
      if (player.translate[0] == object.translate[0] && !(player.translate[2] - 0.15 >= object.translate[2] + 0.15 ||
          player.translate[2] + 0.15 <= object.translate[2] - 0.15)) {
        game_over = true;
      }
    }
    return;
  }

  const home = -2.15, closest = player.translate[2] + 0.45;
  if (object.setback) object.translate[2] = Math.max(closest, object.translate[2] - 0.01);
  else object.translate[2] = Math.min(home, object.translate[2] + 0.005);
}
