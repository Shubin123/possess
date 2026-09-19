/*
 * possess - the built-in pose rules.
 *
 * The trained classifier needs you to record samples first. This head needs
 * nothing: it scores a handful of everyday poses straight from joint angles,
 * so the page classifies something the moment the camera starts, and there is
 * always a reference to compare a freshly trained model against.
 *
 * Each rule is a fuzzy AND (the minimum) of soft constraints on scale-invariant
 * measurements, so scores fade at the edges of a pose instead of snapping, and
 * the measurements themselves are returned for the UI to display.
 */
(function (global) {
  const PZ = (global.PZ = global.PZ || {});
  const F = PZ.features || (typeof require !== 'undefined' ? require('./features.js') : null);
  const LM = F.LM;

  const DEG = 180 / Math.PI;

  function clamp01(x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

  /** 0 at `a`, 1 at `b`, linear in between. Works in either direction. */
  function ramp(x, a, b) { return clamp01((x - a) / (b - a)); }

  /** 1 while within `flat` of `target`, falling to 0 at `flat + fade`. */
  function near(x, target, flat, fade) {
    return clamp01(1 - (Math.abs(x - target) - flat) / fade);
  }

  function fuzzyAnd(scores) { return Math.min.apply(null, scores); }

  /*
   * Everything a rule is allowed to look at: angles in degrees and distances in
   * torso lengths, all independent of where the person stands and how big they
   * appear in frame.
   */
  function measure(featureResult) {
    const norm = featureResult.normalized;
    const p = norm.points;
    const unit = norm.torso / norm.scale || 1e-6;
    const shoulderMid = F.mid(p[LM.L_SHOULDER], p[LM.R_SHOULDER]);
    const ankleMid = F.mid(p[LM.L_ANKLE], p[LM.R_ANKLE]);
    const kneeMid = F.mid(p[LM.L_KNEE], p[LM.R_KNEE]);

    // y grows downwards, so "above" means a smaller y.
    const above = function (a, b) { return (b.y - a.y) / unit; };

    return {
      kneeL: F.angleAt(p[LM.L_HIP], p[LM.L_KNEE], p[LM.L_ANKLE]) * DEG,
      kneeR: F.angleAt(p[LM.R_HIP], p[LM.R_KNEE], p[LM.R_ANKLE]) * DEG,
      elbowL: F.angleAt(p[LM.L_SHOULDER], p[LM.L_ELBOW], p[LM.L_WRIST]) * DEG,
      elbowR: F.angleAt(p[LM.R_SHOULDER], p[LM.R_ELBOW], p[LM.R_WRIST]) * DEG,
      hipL: F.angleAt(p[LM.L_SHOULDER], p[LM.L_HIP], p[LM.L_KNEE]) * DEG,
      hipR: F.angleAt(p[LM.R_SHOULDER], p[LM.R_HIP], p[LM.R_KNEE]) * DEG,
      // 0 = straight up, 90 = horizontal, 180 = straight down.
      armL: Math.abs(F.directionAngle(p[LM.L_SHOULDER], p[LM.L_WRIST]) * DEG),
      armR: Math.abs(F.directionAngle(p[LM.R_SHOULDER], p[LM.R_WRIST]) * DEG),
      torsoLean: Math.abs(F.directionAngle({ x: 0, y: 0 }, shoulderMid) * DEG),
      wristAboveHeadL: above(p[LM.L_WRIST], p[LM.NOSE]),
      wristAboveHeadR: above(p[LM.R_WRIST], p[LM.NOSE]),
      wristAboveShoulderL: above(p[LM.L_WRIST], p[LM.L_SHOULDER]),
      wristAboveShoulderR: above(p[LM.R_WRIST], p[LM.R_SHOULDER]),
      hipAboveKnee: above({ x: 0, y: 0 }, kneeMid),
      bodyExtension: (ankleMid.y - shoulderMid.y) / unit,
      stanceWidth: Math.abs(p[LM.L_ANKLE].x - p[LM.R_ANKLE].x) / unit,
      wristSpan: F.dist(p[LM.L_WRIST], p[LM.R_WRIST]) / unit
    };
  }

  const RULES = [
    {
      label: 'standing',
      describe: 'upright, legs straight, arms down',
      score: function (m) {
        return fuzzyAnd([
          ramp(Math.min(m.kneeL, m.kneeR), 140, 165),
          near(m.torsoLean, 0, 12, 25),
          ramp(Math.min(m.armL, m.armR), 110, 145),
          ramp(m.hipAboveKnee, 0.25, 0.5)
        ]);
      }
    },
    {
      label: 't-pose',
      describe: 'arms straight out to the sides',
      score: function (m) {
        return fuzzyAnd([
          near(m.armL, 90, 15, 25),
          near(m.armR, 90, 15, 25),
          ramp(Math.min(m.elbowL, m.elbowR), 130, 160),
          ramp(m.wristSpan, 1.7, 2.3),
          ramp(m.hipAboveKnee, 0.25, 0.5),
          near(m.torsoLean, 0, 15, 25)
        ]);
      }
    },
    {
      label: 'hands-up',
      describe: 'both hands raised above the head',
      score: function (m) {
        return fuzzyAnd([
          ramp(m.wristAboveHeadL, -0.1, 0.25),
          ramp(m.wristAboveHeadR, -0.1, 0.25),
          ramp(Math.min(m.kneeL, m.kneeR), 120, 155)
        ]);
      }
    },
    {
      label: 'one-arm-up',
      describe: 'one hand raised, the other down',
      score: function (m) {
        const raisedL = fuzzyAnd([
          ramp(m.wristAboveHeadL, -0.1, 0.25),
          ramp(m.wristAboveShoulderR, 0.1, -0.35)
        ]);
        const raisedR = fuzzyAnd([
          ramp(m.wristAboveHeadR, -0.1, 0.25),
          ramp(m.wristAboveShoulderL, 0.1, -0.35)
        ]);
        return Math.max(raisedL, raisedR);
      }
    },
    {
      label: 'squat',
      describe: 'hips dropped towards knee height',
      score: function (m) {
        // Both terms are needed. Hip height alone flags anyone photographed
        // from a low angle, which foreshortens the legs; knee angle alone
        // flags a bent front leg (a lunge, or Warrior II). Measured on
        // reference photos: standing bodies sit at 0.41-0.80 torsos of hip
        // clearance with knees past 160 degrees, a real squat at 0.21 with
        // knees at 86.
        return fuzzyAnd([
          ramp(Math.min(m.kneeL, m.kneeR), 150, 115),
          ramp(m.hipAboveKnee, 0.55, 0.25)
        ]);
      }
    }
  ];

  const LABELS = RULES.map(function (r) { return r.label; });

  /*
   * Scores every rule and returns the same shape the trained heads return, so
   * the page can swap backends without changing how it renders a prediction.
   * Scores are normalized to sum to 1 only when something actually matched;
   * below `minScore` the pose is reported as `unknown`.
   */
  function predict(featureResult, options) {
    if (!featureResult || !featureResult.normalized) return null;
    const opts = options || {};
    const minScore = opts.minScore === undefined ? 0.3 : opts.minScore;
    const m = measure(featureResult);

    const raw = RULES.map(function (rule) { return rule.score(m); });
    const peak = Math.max.apply(null, raw);
    const total = raw.reduce(function (a, b) { return a + b; }, 0);
    const scores = raw.map(function (v) { return total > 1e-6 ? v / total : 0; });

    const prediction = PZ.classifier
      ? PZ.classifier.toPrediction(LABELS, scores)
      : fallbackPrediction(LABELS, scores);
    prediction.measurements = m;
    prediction.peak = peak;
    if (peak < minScore) {
      prediction.label = 'unknown';
      prediction.score = 1 - peak;
    }
    return prediction;
  }

  function fallbackPrediction(labels, scores) {
    const ranked = labels.map(function (label, i) { return { label: label, score: scores[i] }; })
      .sort(function (a, b) { return b.score - a.score; });
    return { label: ranked[0].label, score: ranked[0].score, labels: labels.slice(), scores: scores.slice(), ranked: ranked };
  }

  PZ.rules = {
    LABELS: LABELS,
    RULES: RULES,
    measure: measure,
    predict: predict,
    ramp: ramp,
    near: near
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = PZ.rules;
})(typeof globalThis !== 'undefined' ? globalThis : this);
