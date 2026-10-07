function boost(gl, track, z_dist) {
  return { translate: [track, -0.60, z_dist], rotation: 0, type: getRandomInt(0, 1) == 0 ? "fly" : "jump" };
}

function boost_tick(gl, boosts, player) {
  for (let i = boosts.length - 1; i >= 0; --i) {
    const b = boosts[i];
    b.translate[2] += speed;
    b.rotation += 0.1;
    if (player.translate[0] == b.translate[0] && player.translate[2] - 0.15 <= b.translate[2] &&
        player.translate[2] + 0.15 >= b.translate[2] && grounded(player)) {
      if (b.type == "jump") {
        player.jumpheight = 0.05;
        player.jumpboost = true;
        setTimeout(function() {
          player.jumpheight = -0.15;
          player.jumpboost = false;
        }, 5000);
      } else if (b.type == "fly") {
        player.jumpheight = 0.05;
        player.flyboost = true;
        setTimeout(function() {
          player.flyboost = false;
          if (!player.jumpboost) player.jumpheight = -0.15;
        }, 10000);
      }
      boost_delete(gl, b);
    } else if (b.translate[2] > 2) {
      boost_delete(gl, b);
    }
  }
}

function boost_delete(gl, object) {
  const i = boosts.indexOf(object);
  boosts.splice(i, 1);
}
