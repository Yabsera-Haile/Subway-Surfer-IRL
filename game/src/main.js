var statusKeys = {};
var game_over = false;
var game_start = false;
var finish = 0;
var speed = 0;
var objects = [];
var coins = [];
var obstacles = [];
var barriers = [];
var boosts = [];
var finish_object;
var autopilot = false;

var acc = 0;
var then = null;

function main(quality) {
  objects.push(player(null));
  objects.push(police(null));
  finish_object = finishline(null);
  const canvas = document.querySelector('#glcanvas');
  try {
    Scene.init(canvas, quality);
  } catch (e) {
    graphics_problem("Chrome wouldn't start 3D graphics (" + e.message.replace(/\.$/, "") + "). This usually follows a GPU " +
      "reset: restart Chrome. If it keeps happening, check chrome://gpu for WebGL.");
    return;
  }
  canvas.addEventListener('webglcontextlost', function() {
    graphics_problem("The GPU reset and the game lost its graphics, which can happen when the camera model " +
      "and the game load it at the same time. Reload. If it happens again, choose Low graphics, or set " +
      "the extension's Pose engine to CPU.");
  });
  requestAnimationFrame(render);
}

function render(nowMs) {
  const now = nowMs / 1000;
  const deltaTime = then == null ? 0 : now - then;
  then = now;

  update_score();
  if (game_over || finish) {
    Game_over();
  }

  if (!game_over && !finish && game_start) {
    acc += Math.min(deltaTime, 0.25);
    while (acc >= STEP) {
      acc -= STEP;
      if (autopilot) autopilot_tick();
      step(null);
    }
  }

  Scene.frame(Math.min(deltaTime, 0.1));
  requestAnimationFrame(render);
}
