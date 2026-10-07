function barrier(gl, track, z_dist) {
  return { translate: [track, -0.78, z_dist], initial_z: z_dist, type: "barricade" };
}

function barrier_tick(gl, barriers, player, police) {
  for (let i = barriers.length - 1; i >= 0; --i) {
    const b = barriers[i];
    b.translate[2] += speed;
    if (!b.resolved && b.contactAt == null && player.translate[0] == b.translate[0] &&
        zOverlap(player, b, 0.05) && grounded(player)) {
      b.contactAt = simTime;
    }
    if (!b.resolved && b.contactAt != null) {
      if (!grounded(player) || player.translate[0] != b.translate[0]) {
        b.resolved = true;
        saves++;
      } else if (simTime - b.contactAt >= CONFIG.graceS) {
        b.resolved = true;
        stumble(police);
      }
    }
    if (b.translate[2] > 2) barrier_delete(gl, b);
  }
}

function barrier_delete(gl, object) {
  const i = barriers.indexOf(object);
  barriers.splice(i, 1);
}
