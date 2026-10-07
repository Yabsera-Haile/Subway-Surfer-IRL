function coin(gl, initial_x, initial_z) {
  return { translate: [initial_x, -0.75, initial_z], rotation: 0, type: "coins" };
}

function coin_tick(gl, coins, player) {
  for (let i = coins.length - 1; i >= 0; --i) {
    const c = coins[i];
    c.translate[2] += speed;
    c.rotation -= 0.1;
    if (player.translate[0] == c.translate[0] && player.translate[2] - 0.15 <= c.translate[2] &&
        player.translate[2] + 0.15 >= c.translate[2] && grounded(player)) {
      player.score += 1;
      coin_delete(gl, c);
    } else if (c.translate[2] > 2) {
      coin_delete(gl, c);
    }
  }
}

function coin_delete(gl, object) {
  const i = coins.indexOf(object);
  coins.splice(i, 1);
}
