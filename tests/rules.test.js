/*
 * The built-in rules are the zero-setup classifier, so every pose they claim to
 * know has to come out on top - including when the figure is small, off-centre
 * or in a widescreen frame.
 */
const test = require('node:test');
const assert = require('node:assert/strict');

const features = require('../features.js');
require('../classifier.js');
const rules = require('../rules.js');
const { makeLandmarks } = require('./fixtures.js');

function classify(pose, options) {
  const feature = features.featurize(makeLandmarks(pose, options), { aspect: 1 });
  return rules.predict(feature);
}

const CASES = [
  ['standing', 'standing'],
  ['t-pose', 't-pose'],
  ['hands-up', 'hands-up'],
  ['one-arm-up-left', 'one-arm-up'],
  ['one-arm-up-right', 'one-arm-up'],
  ['squat', 'squat']
];

CASES.forEach(function (pair) {
  test('recognizes ' + pair[0], () => {
    const prediction = classify(pair[0]);
    assert.equal(prediction.label, pair[1]);
    assert.ok(prediction.peak > 0.8, 'confident: ' + prediction.peak.toFixed(2));
  });

  test('recognizes ' + pair[0] + ' anywhere in the frame', () => {
    const prediction = classify(pair[0], { scale: 0.45, cx: 0.28, cy: 0.58 });
    assert.equal(prediction.label, pair[1]);
  });
});

test('every rule scores its own pose highest', () => {
  CASES.forEach(function (pair) {
    const prediction = classify(pair[0]);
    prediction.ranked.slice(1).forEach(function (other) {
      assert.ok(prediction.ranked[0].score > other.score + 0.05,
        pair[0] + ' beats ' + other.label);
    });
  });
});

test('reports unknown rather than guessing', () => {
  // Arms halfway to a T-pose, hips halfway to a squat: deliberately nothing.
  const landmarks = makeLandmarks('standing');
  const feature = features.featurize(landmarks, { aspect: 1 });
  const measured = rules.measure(feature);
  assert.ok(measured.kneeL > 150, 'the fixture really is standing');

  const ambiguous = makeLandmarks('squat', { scale: 0.8 });
  // Raise the hips halfway back up towards standing.
  [23, 24].forEach(function (i) { ambiguous[i].y -= 0.08; });
  const prediction = rules.predict(features.featurize(ambiguous, { aspect: 1 }));
  assert.ok(prediction.peak < 0.8, 'a half-squat is not confidently anything');
});

test('measurements are reported in degrees and torso lengths', () => {
  const m = rules.measure(features.featurize(makeLandmarks('t-pose'), { aspect: 1 }));
  assert.ok(Math.abs(m.armL - 90) < 5, 'a T-pose has horizontal arms');
  assert.ok(m.kneeL > 165 && m.kneeL <= 180, 'straight legs');
  assert.ok(m.hipAboveKnee > 0.6, 'hips well above the knees when standing');
});
