const DIFFICULTIES = {
  gesture: {
    label: "Gesture (Kinect-style)",
    speed: 3.0, maxSpeed: 4.0, rampPerMin: 0.5,
    gapS: [1.8, 2.6], hazardsPerRow: 1, trainShare: 0.55,
    graceS: 0.30, jumpS: 0.80,
    chase: "forgiving", chaseS: 6, stumbleSlow: 0.6, stumbleSlowS: 1.0,
  },
  normal: {
    label: "Normal",
    speed: 4.0, maxSpeed: 5.0, rampPerMin: 0.5,
    gapS: [1.2, 1.8], hazardsPerRow: 2, trainShare: 0.55,
    graceS: 0.15, jumpS: 0.60,
    chase: "forgiving", chaseS: 6, stumbleSlow: 0.7, stumbleSlowS: 1.0,
  },
  classic: {
    label: "Classic (close to the original)",
    speed: 4.5, maxSpeed: 4.5, rampPerMin: 0,
    gapS: [0.7, 1.3], hazardsPerRow: 2, trainShare: 0.5,
    graceS: 0, jumpS: 0.30,
    chase: "classic", chaseS: 7, stumbleSlow: 0.9, stumbleSlowS: 7,
  },
};

var CONFIG = DIFFICULTIES.gesture;

function setDifficulty(name) {
  CONFIG = DIFFICULTIES[name] || DIFFICULTIES.gesture;
}
