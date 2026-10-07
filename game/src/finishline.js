function finishline(gl) {
  return { translate: [0, 1.5, -45], type: "finishline" };
}

function finishline_tick(flag, player){
  flag.translate[2] += speed;

  if(!(player.translate[2] - 0.15 >= flag.translate[2] + 0.1 || player.translate[2] + 0.15 <= flag.translate[2] - 0.1)){
    finish = 1;
  }
}
