/*
 * The feature vector has one job: describe the posture and nothing else.
 * These tests pin down the invariances the classifier depends on.
 */
const test = require('node:test');
const assert = require('node:assert/strict');

const features = require('../features.js');
const { makeLandmarks } = require('./fixtures.js');

function values(landmarks, aspect) {
  const result = features.featurize(landmarks, { aspect: aspect === undefined ? 1 : aspect });
  assert.ok(result, 'featurize returned a result');
  return result.values;
}

function maxDifference(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

/** Squeezes x towards the centre, the way a wide frame normalizes it. */
function squeeze(landmarks, aspect) {
  return landmarks.map(function (p) {
    return { x: 0.5 + (p.x - 0.5) / aspect, y: p.y, z: p.z, visibility: p.visibility };
  });
}

test('produces one finite value per named feature', () => {
  const result = features.featurize(makeLandmarks('t-pose'), { aspect: 1 });
  assert.equal(result.values.length, features.FEATURE_LENGTH);
  assert.equal(features.FEATURE_NAMES.length, features.FEATURE_LENGTH);
  assert.ok(Array.from(result.values).every(Number.isFinite), 'no NaN or Infinity');
});

test('ignores where in the frame the person stands and how big they are', () => {
  const near = values(makeLandmarks('t-pose', { scale: 0.85, cx: 0.5, cy: 0.5 }));
  const far = values(makeLandmarks('t-pose', { scale: 0.4, cx: 0.22, cy: 0.63 }));
  assert.ok(maxDifference(near, far) < 1e-4, 'same pose, same vector');
});

test('ignores the aspect ratio of the frame', () => {
  const square = values(makeLandmarks('standing'));
  const widescreen = values(squeeze(makeLandmarks('standing'), 16 / 9), 16 / 9);
  assert.ok(maxDifference(square, widescreen) < 1e-4, 'aspect correction undoes the stretch');
});

test('mirroring swaps left and right', () => {
  const mirroredLeft = values(features.mirrorLandmarks(makeLandmarks('one-arm-up-left')));
  const right = values(makeLandmarks('one-arm-up-right'));
  assert.ok(maxDifference(mirroredLeft, right) < 1e-4, 'a mirrored left-arm raise is a right-arm raise');
});

test('different poses produce different vectors', () => {
  const standing = values(makeLandmarks('standing'));
  const tPose = values(makeLandmarks('t-pose'));
  assert.ok(maxDifference(standing, tPose) > 0.2, 'standing and a T-pose are far apart');
});

test('reports when too much of the body is out of frame', () => {
  const clear = features.featurize(makeLandmarks('standing', { visibility: 0.95 }), { aspect: 1 });
  const obscured = features.featurize(makeLandmarks('standing', { visibility: 0.2 }), { aspect: 1 });
  assert.equal(clear.visible, true);
  assert.equal(obscured.visible, false);
  assert.ok(obscured.values.length === features.FEATURE_LENGTH, 'the vector is still produced');
});

test('refuses input that is not a full skeleton', () => {
  assert.equal(features.featurize(null, {}), null);
  assert.equal(features.featurize(makeLandmarks('standing').slice(0, 20), {}), null);
});

test('collapsed landmarks do not divide by zero', () => {
  const flat = new Array(33).fill(null).map(function () { return { x: 0.5, y: 0.5, visibility: 1 }; });
  assert.equal(features.featurize(flat, { aspect: 1 }), null);
});
