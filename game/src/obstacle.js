function obstacle(gl, track, z_dist) {
  return { translate: [track, -0.60, z_dist], initial_z: z_dist, type: "obstacle" };
}

function obstacle_tick(gl, obstacles, player) {
  for (let i = obstacles.length - 1; i >= 0; --i) {
    const o = obstacles[i];
    o.translate[2] += speed;
    const inPath = player.translate[0] == o.translate[0] && zOverlap(player, o, 0.75) &&
      player.translate[1] < o.translate[1] + 0.5;
    if (inPath) {
      if (o.hitSince == null) o.hitSince = simTime;
      if (simTime - o.hitSince >= CONFIG.graceS) game_over = true;
    } else if (o.hitSince != null) {
      o.hitSince = null;
      saves++;
    }
    if (o.translate[2] > 2) obstacle_delete(gl, o);
  }
}

function obstacle_delete(gl, object) {
  const i = obstacles.indexOf(object);
  obstacles.splice(i, 1);
}
