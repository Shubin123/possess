/*
 * Synthetic BlazePose skeletons for the test suite.
 *
 * Landmarks are built from anthropometric fractions of standing height, laid
 * out the way MediaPipe reports them: 33 points, x/y normalized to the frame,
 * y increasing downwards, and "left" meaning the subject's left.
 *
 * Having a generator rather than captured JSON means the tests can vary the
 * pose, the position in frame, the apparent size and the aspect ratio, which is
 * exactly what the normalization in features.js is supposed to absorb.
 */
const { LM } = require('../features.js');
const { mulberry32 } = require('../classifier.js');

// Heights as fractions of total standing height, measured from the floor.
const BODY = {
  hipY: 0.53, hipX: 0.05,
  shoulderY: 0.82, shoulderX: 0.10,
  noseY: 0.93,
  kneeY: 0.285, kneeX: 0.055,
  ankleY: 0.045, ankleX: 0.05,
  upperArm: 0.17, foreArm: 0.16
};

// Arm elevation in degrees: 0 points straight up, 90 straight out, 180 down.
const POSES = {
  standing: { armL: 172, armR: 172 },
  't-pose': { armL: 90, armR: 90 },
  'hands-up': { armL: 8, armR: 8 },
  'one-arm-up-left': { armL: 5, armR: 172 },
  'one-arm-up-right': { armL: 172, armR: 5 },
  squat: {
    armL: 148, armR: 148,
    hipY: 0.33, shoulderY: 0.60, noseY: 0.70, kneeY: 0.30, kneeX: 0.09
  }
};

/*
 * Builds one 33-landmark frame.
 *
 *   pose     name from POSES
 *   scale    apparent height of the figure as a fraction of frame height
 *   cx, cy   where the figure sits in the frame
 *   jitter   noise in frame units, applied per landmark
 *   seed     PRNG seed, so a test can ask for the same noise twice
 */
function makeLandmarks(pose, options) {
  const opts = options || {};
  const spec = Object.assign({}, BODY, POSES[pose]);
  if (!POSES[pose]) throw new Error('Unknown fixture pose: ' + pose);

  const scale = opts.scale === undefined ? 0.8 : opts.scale;
  const cx = opts.cx === undefined ? 0.5 : opts.cx;
  const cy = opts.cy === undefined ? 0.5 : opts.cy;
  const jitter = opts.jitter || 0;
  const rand = opts.rand || mulberry32(opts.seed === undefined ? 7 : opts.seed);
  const visibility = opts.visibility === undefined ? 1 : opts.visibility;

  // Body units (y up, hip-centred) into frame units (y down, centred on cx/cy).
  const centreY = (spec.shoulderY + spec.ankleY) / 2;
  function place(bx, by) {
    return { x: cx + bx * scale, y: cy - (by - centreY) * scale };
  }

  const points = new Array(33);
  const set = function (index, bx, by) { points[index] = place(bx, by); };

  // Sides: the subject's left appears at larger x in an unmirrored frame.
  const side = { L: 1, R: -1 };
  const shoulders = {}, wrists = {};
  ['L', 'R'].forEach(function (s) {
    const sx = side[s] * spec.shoulderX;
    shoulders[s] = { x: sx, y: spec.shoulderY };
    const elevation = (s === 'L' ? spec.armL : spec.armR) * Math.PI / 180;
    const dx = Math.sin(elevation) * side[s];
    const dy = Math.cos(elevation);
    const elbow = { x: sx + dx * spec.upperArm, y: spec.shoulderY + dy * spec.upperArm };
    const wrist = { x: elbow.x + dx * spec.foreArm, y: elbow.y + dy * spec.foreArm };
    wrists[s] = wrist;
    set(s === 'L' ? LM.L_SHOULDER : LM.R_SHOULDER, sx, spec.shoulderY);
    set(s === 'L' ? LM.L_ELBOW : LM.R_ELBOW, elbow.x, elbow.y);
    set(s === 'L' ? LM.L_WRIST : LM.R_WRIST, wrist.x, wrist.y);
    // Hands: pinky, index and thumb trail just past the wrist.
    const hx = wrist.x + dx * 0.03, hy = wrist.y + dy * 0.03;
    set(s === 'L' ? 17 : 18, hx, hy);
    set(s === 'L' ? 19 : 20, hx + dx * 0.01, hy + dy * 0.01);
    set(s === 'L' ? 21 : 22, wrist.x + dx * 0.02, wrist.y + dy * 0.02);

    set(s === 'L' ? LM.L_HIP : LM.R_HIP, side[s] * spec.hipX, spec.hipY);
    set(s === 'L' ? LM.L_KNEE : LM.R_KNEE, side[s] * spec.kneeX, spec.kneeY);
    set(s === 'L' ? LM.L_ANKLE : LM.R_ANKLE, side[s] * spec.ankleX, spec.ankleY);
    set(s === 'L' ? 29 : 30, side[s] * spec.ankleX - side[s] * 0.01, spec.ankleY - 0.03); // heel
    set(s === 'L' ? LM.L_FOOT : LM.R_FOOT, side[s] * spec.ankleX + side[s] * 0.02, spec.ankleY - 0.04);
  });

  set(LM.NOSE, 0, spec.noseY);
  set(1, 0.012, spec.noseY + 0.015); set(2, 0.02, spec.noseY + 0.015); set(3, 0.028, spec.noseY + 0.015);
  set(4, -0.012, spec.noseY + 0.015); set(5, -0.02, spec.noseY + 0.015); set(6, -0.028, spec.noseY + 0.015);
  set(7, 0.045, spec.noseY + 0.01); set(8, -0.045, spec.noseY + 0.01);
  set(9, 0.018, spec.noseY - 0.02); set(10, -0.018, spec.noseY - 0.02);

  return points.map(function (p) {
    return {
      x: p.x + (jitter ? (rand() * 2 - 1) * jitter : 0),
      y: p.y + (jitter ? (rand() * 2 - 1) * jitter : 0),
      z: 0,
      visibility: visibility
    };
  });
}

/** A labelled batch: `count` jittered frames per pose. */
function makeDataset(poses, count, options) {
  const opts = options || {};
  const rand = mulberry32(opts.seed === undefined ? 11 : opts.seed);
  const frames = [];
  poses.forEach(function (pose, target) {
    for (let i = 0; i < count; i++) {
      frames.push({
        pose: pose,
        target: target,
        landmarks: makeLandmarks(pose, {
          rand: rand,
          jitter: opts.jitter === undefined ? 0.01 : opts.jitter,
          scale: 0.7 + rand() * 0.25,
          cx: 0.35 + rand() * 0.3,
          cy: 0.4 + rand() * 0.2
        })
      });
    }
  });
  return frames;
}

module.exports = { makeLandmarks, makeDataset, POSES, BODY };
