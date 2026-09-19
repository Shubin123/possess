/*
 * poser - landmarks to a pose feature vector (no frameworks, no build step).
 *
 * The pose estimator (MediaPipe Pose Landmarker) returns 33 landmarks in
 * normalized image coordinates.  Raw landmarks are a poor classifier input:
 * they move when the person walks across the frame, grow when they step
 * towards the camera, and stretch with the aspect ratio of the video.
 *
 * This file turns them into a vector that only describes *posture*:
 *
 *   1. x is multiplied by the image aspect ratio so geometry is isotropic,
 *   2. the hip midpoint becomes the origin,
 *   3. everything is divided by a torso-derived scale (the normalization used
 *      by Google's MoveNet pose-classification recipe),
 *   4. joint angles, limb directions and a few body ratios are appended,
 *      because those are already translation/scale invariant and give a small
 *      classifier a much easier job than bare coordinates.
 *
 * Pure and dependency-free so it runs identically in the browser and in
 * `node --test`.
 */
(function (global) {
  const PZ = (global.PZ = global.PZ || {});

  // MediaPipe Pose Landmarker indices (BlazePose GHUM, 33 points).
  const LM = {
    NOSE: 0,
    L_SHOULDER: 11, R_SHOULDER: 12,
    L_ELBOW: 13, R_ELBOW: 14,
    L_WRIST: 15, R_WRIST: 16,
    L_HIP: 23, R_HIP: 24,
    L_KNEE: 25, R_KNEE: 26,
    L_ANKLE: 27, R_ANKLE: 28,
    L_FOOT: 31, R_FOOT: 32
  };

  // The points that carry posture. The dense face and hand landmarks are
  // dropped: they add 36 noisy dimensions and say nothing about the pose.
  const POINTS = [
    LM.NOSE,
    LM.L_SHOULDER, LM.R_SHOULDER,
    LM.L_ELBOW, LM.R_ELBOW,
    LM.L_WRIST, LM.R_WRIST,
    LM.L_HIP, LM.R_HIP,
    LM.L_KNEE, LM.R_KNEE,
    LM.L_ANKLE, LM.R_ANKLE,
    LM.L_FOOT, LM.R_FOOT
  ];

  const POINT_NAMES = [
    'nose',
    'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
    'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle', 'l_foot', 'r_foot'
  ];

  // Angle at the middle joint of each triple.
  const ANGLES = [
    { name: 'l_elbow', a: LM.L_SHOULDER, b: LM.L_ELBOW, c: LM.L_WRIST },
    { name: 'r_elbow', a: LM.R_SHOULDER, b: LM.R_ELBOW, c: LM.R_WRIST },
    { name: 'l_shoulder', a: LM.L_ELBOW, b: LM.L_SHOULDER, c: LM.L_HIP },
    { name: 'r_shoulder', a: LM.R_ELBOW, b: LM.R_SHOULDER, c: LM.R_HIP },
    { name: 'l_hip', a: LM.L_SHOULDER, b: LM.L_HIP, c: LM.L_KNEE },
    { name: 'r_hip', a: LM.R_SHOULDER, b: LM.R_HIP, c: LM.R_KNEE },
    { name: 'l_knee', a: LM.L_HIP, b: LM.L_KNEE, c: LM.L_ANKLE },
    { name: 'r_knee', a: LM.R_HIP, b: LM.R_KNEE, c: LM.R_ANKLE }
  ];

  // Limb directions, measured against straight up and encoded as cos/sin so
  // the classifier never sees the -pi/+pi discontinuity.
  const DIRECTIONS = [
    { name: 'torso', from: 'hipMid', to: 'shoulderMid' },
    { name: 'l_arm', from: LM.L_SHOULDER, to: LM.L_WRIST },
    { name: 'r_arm', from: LM.R_SHOULDER, to: LM.R_WRIST },
    { name: 'l_leg', from: LM.L_HIP, to: LM.L_ANKLE },
    { name: 'r_leg', from: LM.R_HIP, to: LM.R_ANKLE }
  ];

  const RATIO_NAMES = ['stance_width', 'body_extension', 'wrist_span', 'knee_drop'];

  // Left/right index pairs, used when mirroring a sample.
  const MIRROR_PAIRS = [
    [1, 4], [2, 5], [3, 6],         // eyes
    [7, 8], [9, 10],                // ears, mouth
    [11, 12], [13, 14], [15, 16],   // shoulders, elbows, wrists
    [17, 18], [19, 20], [21, 22],   // pinkies, index fingers, thumbs
    [23, 24], [25, 26], [27, 28],   // hips, knees, ankles
    [29, 30], [31, 32]              // heels, foot indices
  ];

  // Bones, for drawing the skeleton overlay.
  const SKELETON = [
    [LM.L_SHOULDER, LM.R_SHOULDER], [LM.L_SHOULDER, LM.L_ELBOW],
    [LM.L_ELBOW, LM.L_WRIST], [LM.R_SHOULDER, LM.R_ELBOW],
    [LM.R_ELBOW, LM.R_WRIST], [LM.L_SHOULDER, LM.L_HIP],
    [LM.R_SHOULDER, LM.R_HIP], [LM.L_HIP, LM.R_HIP],
    [LM.L_HIP, LM.L_KNEE], [LM.L_KNEE, LM.L_ANKLE],
    [LM.R_HIP, LM.R_KNEE], [LM.R_KNEE, LM.R_ANKLE],
    [LM.L_ANKLE, LM.L_FOOT], [LM.R_ANKLE, LM.R_FOOT]
  ];

  const FEATURE_NAMES = (function () {
    const names = [];
    POINT_NAMES.forEach(function (p) { names.push(p + '_x', p + '_y'); });
    ANGLES.forEach(function (a) { names.push(a.name + '_cos', a.name + '_sin'); });
    DIRECTIONS.forEach(function (d) { names.push(d.name + '_dir_cos', d.name + '_dir_sin'); });
    RATIO_NAMES.forEach(function (r) { names.push(r); });
    return names;
  })();

  const FEATURE_LENGTH = FEATURE_NAMES.length;

  function mid(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  }

  function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  /** Interior angle at `b`, in radians, 0..pi. */
  function angleAt(a, b, c) {
    const v1x = a.x - b.x, v1y = a.y - b.y;
    const v2x = c.x - b.x, v2y = c.y - b.y;
    const n1 = Math.hypot(v1x, v1y), n2 = Math.hypot(v2x, v2y);
    if (n1 < 1e-9 || n2 < 1e-9) return 0;
    const cos = (v1x * v2x + v1y * v2y) / (n1 * n2);
    return Math.acos(Math.min(1, Math.max(-1, cos)));
  }

  /** Direction of b-a measured clockwise from straight up, in radians. */
  function directionAngle(a, b) {
    return Math.atan2(b.x - a.x, -(b.y - a.y));
  }

  /** Mean of the `visibility` scores of the posture landmarks. */
  function meanVisibility(landmarks) {
    let sum = 0;
    for (let i = 0; i < POINTS.length; i++) {
      const v = landmarks[POINTS[i]] && landmarks[POINTS[i]].visibility;
      sum += (v === undefined || v === null) ? 1 : v;
    }
    return sum / POINTS.length;
  }

  /*
   * Aspect-corrects the landmarks, recenters them on the hips and divides by a
   * torso-derived scale. Returns the normalized posture points plus the
   * intermediate geometry the feature builder and the rule classifier reuse.
   *
   * `scale` follows Google's recipe: the larger of (torso length x 2.5) and the
   * distance to the furthest posture point, which keeps a curled-up pose from
   * being blown up by its own short torso.
   */
  function normalize(landmarks, options) {
    const opts = options || {};
    const aspect = opts.aspect || 1;
    const torsoMultiplier = opts.torsoMultiplier || 2.5;

    const pts = landmarks.map(function (p) {
      return { x: p.x * aspect, y: p.y, visibility: p.visibility };
    });
    const hipMid = mid(pts[LM.L_HIP], pts[LM.R_HIP]);
    const shoulderMid = mid(pts[LM.L_SHOULDER], pts[LM.R_SHOULDER]);
    const torso = dist(hipMid, shoulderMid);

    let furthest = 0;
    for (let i = 0; i < POINTS.length; i++) {
      furthest = Math.max(furthest, dist(pts[POINTS[i]], hipMid));
    }
    const scale = Math.max(torso * torsoMultiplier, furthest);
    if (!(scale > 1e-6)) return null;

    const centered = pts.map(function (p) {
      return { x: (p.x - hipMid.x) / scale, y: (p.y - hipMid.y) / scale, visibility: p.visibility };
    });
    return { points: centered, raw: pts, hipMid: hipMid, shoulderMid: shoulderMid, torso: torso, scale: scale };
  }

  /*
   * Builds the classifier input for one detected pose.
   *
   * Returns null when the landmarks are unusable (degenerate scale), and sets
   * `visible: false` when too much of the body is out of frame - callers use
   * that to refuse to record a training sample or to trust a prediction.
   */
  function featurize(landmarks, options) {
    if (!landmarks || landmarks.length < 33) return null;
    const opts = options || {};
    const norm = normalize(landmarks, opts);
    if (!norm) return null;

    const p = norm.points;
    const values = new Float32Array(FEATURE_LENGTH);
    let at = 0;

    for (let i = 0; i < POINTS.length; i++) {
      values[at++] = p[POINTS[i]].x;
      values[at++] = p[POINTS[i]].y;
    }
    for (let i = 0; i < ANGLES.length; i++) {
      const a = ANGLES[i];
      const theta = angleAt(p[a.a], p[a.b], p[a.c]);
      values[at++] = Math.cos(theta);
      values[at++] = Math.sin(theta);
    }
    const hipOrigin = { x: 0, y: 0 };
    const shoulderMid = mid(p[LM.L_SHOULDER], p[LM.R_SHOULDER]);
    for (let i = 0; i < DIRECTIONS.length; i++) {
      const d = DIRECTIONS[i];
      const from = d.from === 'hipMid' ? hipOrigin : p[d.from];
      const to = d.to === 'shoulderMid' ? shoulderMid : p[d.to];
      const theta = directionAngle(from, to);
      values[at++] = Math.cos(theta);
      values[at++] = Math.sin(theta);
    }

    const ankleMid = mid(p[LM.L_ANKLE], p[LM.R_ANKLE]);
    const kneeMid = mid(p[LM.L_KNEE], p[LM.R_KNEE]);
    const unit = norm.torso / norm.scale || 1e-6; // torso length in normalized units
    values[at++] = Math.abs(p[LM.L_ANKLE].x - p[LM.R_ANKLE].x) / unit;
    values[at++] = (ankleMid.y - shoulderMid.y) / unit;
    values[at++] = dist(p[LM.L_WRIST], p[LM.R_WRIST]) / unit;
    values[at++] = kneeMid.y / unit;

    const visibility = meanVisibility(landmarks);
    return {
      values: values,
      visibility: visibility,
      visible: visibility >= (opts.minVisibility === undefined ? 0.5 : opts.minVisibility),
      normalized: norm
    };
  }

  /*
   * Mirrors a pose left-to-right. Recording a sample stores both the pose and
   * its mirror, so a classifier trained on someone raising their right arm also
   * recognizes the left arm and a mirrored webcam feed.
   */
  function mirrorLandmarks(landmarks) {
    const out = landmarks.map(function (p) {
      return { x: 1 - p.x, y: p.y, z: p.z, visibility: p.visibility };
    });
    for (let i = 0; i < MIRROR_PAIRS.length; i++) {
      const a = MIRROR_PAIRS[i][0], b = MIRROR_PAIRS[i][1];
      if (a < out.length && b < out.length) {
        const tmp = out[a]; out[a] = out[b]; out[b] = tmp;
      }
    }
    return out;
  }

  PZ.features = {
    LM: LM,
    POINTS: POINTS,
    POINT_NAMES: POINT_NAMES,
    ANGLES: ANGLES,
    SKELETON: SKELETON,
    FEATURE_NAMES: FEATURE_NAMES,
    FEATURE_LENGTH: FEATURE_LENGTH,
    featurize: featurize,
    normalize: normalize,
    mirrorLandmarks: mirrorLandmarks,
    meanVisibility: meanVisibility,
    angleAt: angleAt,
    directionAngle: directionAngle,
    mid: mid,
    dist: dist
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = PZ.features;
})(typeof globalThis !== 'undefined' ? globalThis : this);
