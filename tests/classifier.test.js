/*
 * The classifier heads, trained on synthetic skeletons: a real dataset would
 * make these tests slower and no more informative, because what is under test
 * is the learning and the bookkeeping, not the anatomy.
 */
const test = require('node:test');
const assert = require('node:assert/strict');

const features = require('../features.js');
const { KnnClassifier, MlpClassifier, PredictionSmoother, stratifiedSplit, mulberry32, softmax, argmax } =
  require('../classifier.js');
const { makeLandmarks, makeDataset } = require('./fixtures.js');

const POSES = ['standing', 't-pose', 'hands-up', 'squat'];

function buildTraining(count, seed) {
  const frames = makeDataset(POSES, count, { seed: seed, jitter: 0.012 });
  const rows = [], targets = [];
  frames.forEach(function (frame) {
    const feature = features.featurize(frame.landmarks, { aspect: 1 });
    if (!feature) return;
    rows.push(feature.values);
    targets.push(frame.target);
  });
  return { rows: rows, targets: targets, labels: POSES.slice() };
}

function featuresFor(pose) {
  return features.featurize(makeLandmarks(pose, { scale: 0.62, cx: 0.42, cy: 0.46 }), { aspect: 1 }).values;
}

test('softmax is a probability distribution and argmax finds its peak', () => {
  const probs = softmax([2, 1, 0.1, -5]);
  const total = probs.reduce(function (a, b) { return a + b; }, 0);
  assert.ok(Math.abs(total - 1) < 1e-6);
  assert.equal(argmax(probs), 0);
  // Large logits must not overflow to NaN.
  assert.ok(softmax([1000, 999]).every(Number.isFinite));
});

test('k-nearest-neighbours classifies straight after recording', () => {
  const data = buildTraining(20, 3);
  const knn = new KnnClassifier({ k: 5 }).fit(data.rows, data.targets, data.labels);
  POSES.forEach(function (pose) {
    const prediction = knn.predict(featuresFor(pose));
    assert.equal(prediction.label, pose);
    assert.ok(prediction.score > 0.5, pose + ' scored ' + prediction.score.toFixed(2));
    assert.equal(prediction.scores.length, POSES.length);
  });
});

test('the network trains to high held-out accuracy', () => {
  const data = buildTraining(60, 5);
  const model = new MlpClassifier({ hidden: 32 });
  const report = model.fit(data.rows, data.targets, data.labels, { epochs: 120, seed: 1337 });
  assert.ok(report.trainAccuracy > 0.95, 'train accuracy ' + report.trainAccuracy);
  assert.ok(report.valAccuracy > 0.9, 'validation accuracy ' + report.valAccuracy);
  assert.ok(report.validationSamples > 0, 'something was held out');
  assert.ok(report.loss < 0.3, 'loss came down to ' + report.loss.toFixed(3));
  POSES.forEach(function (pose) {
    assert.equal(model.predict(featuresFor(pose)).label, pose);
  });
});

test('training is deterministic for a given seed', () => {
  const data = buildTraining(20, 9);
  const a = new MlpClassifier({ hidden: 16 });
  const b = new MlpClassifier({ hidden: 16 });
  const reportA = a.fit(data.rows, data.targets, data.labels, { epochs: 20, seed: 42 });
  const reportB = b.fit(data.rows, data.targets, data.labels, { epochs: 20, seed: 42 });
  assert.equal(reportA.loss, reportB.loss);
  assert.deepEqual(a.predict(featuresFor('t-pose')).scores, b.predict(featuresFor('t-pose')).scores);
});

test('an exported model predicts exactly like the one it came from', () => {
  const data = buildTraining(25, 11);
  const model = new MlpClassifier({ hidden: 24 });
  model.fit(data.rows, data.targets, data.labels, { epochs: 40, seed: 7 });

  const json = JSON.parse(JSON.stringify(model.toJSON()));
  const restored = MlpClassifier.fromJSON(json);
  assert.deepEqual(restored.labels, model.labels);
  POSES.forEach(function (pose) {
    const row = featuresFor(pose);
    assert.deepEqual(restored.predict(row).scores, model.predict(row).scores);
  });
});

test('refuses to load something that is not a poser model', () => {
  assert.throws(() => MlpClassifier.fromJSON({ format: 'something-else' }), /not a poser model/i);
  assert.throws(() => MlpClassifier.fromJSON(null), /not a poser model/i);
  assert.throws(() => new MlpClassifier().toJSON(), /not trained/i);
});

test('refuses to train on a single class', () => {
  const data = buildTraining(12, 13);
  assert.throws(
    () => new MlpClassifier().fit(data.rows, data.targets.map(() => 0), ['only-one'], { epochs: 1 }),
    /at least two classes/i
  );
});

test('the validation split holds out every class and starves none', () => {
  const targets = [];
  for (let i = 0; i < 40; i++) targets.push(0);
  for (let i = 0; i < 7; i++) targets.push(1);
  targets.push(2); // a class with a single sample
  const split = stratifiedSplit(targets, 3, 0.2, mulberry32(5));
  [0, 1, 2].forEach(function (c) {
    assert.ok(split.train.some(function (i) { return targets[i] === c; }), 'class ' + c + ' is trained on');
  });
  assert.ok(split.val.some(function (i) { return targets[i] === 0; }), 'the big class is validated');
  assert.equal(split.train.length + split.val.length, targets.length);
});

/* --- smoothing and repetition counting --- */

function prediction(labels, scores) {
  const ranked = labels.map(function (label, i) { return { label: label, score: scores[i] }; })
    .sort(function (a, b) { return b.score - a.score; });
  return { label: ranked[0].label, score: ranked[0].score, labels: labels.slice(), scores: scores, ranked: ranked };
}

test('a single noisy frame does not flip the reported pose', () => {
  const smoother = new PredictionSmoother({ alpha: 0.35, threshold: 0.6, holdFrames: 3 });
  const labels = ['standing', 'squat'];
  for (let i = 0; i < 12; i++) smoother.push(prediction(labels, [0.95, 0.05]));
  assert.equal(smoother.current().label, 'standing');
  smoother.push(prediction(labels, [0.1, 0.9])); // one bad frame
  assert.equal(smoother.current().label, 'standing', 'still standing');
});

test('counts a repetition each time a pose is held', () => {
  const smoother = new PredictionSmoother({ alpha: 0.5, threshold: 0.6, holdFrames: 3 });
  const labels = ['standing', 'squat'];
  const hold = function (index, frames) {
    const scores = index === 0 ? [0.98, 0.02] : [0.02, 0.98];
    for (let i = 0; i < frames; i++) smoother.push(prediction(labels, scores));
  };
  hold(0, 10); hold(1, 10); hold(0, 10); hold(1, 10); hold(0, 10);
  const counts = smoother.current().counts;
  assert.equal(counts.squat, 2, 'two squats');
  assert.equal(counts.standing, 3, 'three times back to standing');
});

test('starts the average over when the class list changes', () => {
  const smoother = new PredictionSmoother({ holdFrames: 3 });
  for (let i = 0; i < 5; i++) smoother.push(prediction(['a', 'b'], [0.9, 0.1]));
  assert.equal(smoother.current().label, 'a');
  const after = smoother.push(prediction(['x', 'y', 'z'], [0.1, 0.8, 0.1]));
  assert.equal(after.scores.length, 3, 'the old two-class average was discarded');
  assert.equal(after.label, null, 'no label is claimed on the first frame of a new model');
});

test('losing the person clears the candidate', () => {
  const smoother = new PredictionSmoother({ holdFrames: 3, threshold: 0.6 });
  const labels = ['standing', 'squat'];
  smoother.push(prediction(labels, [0.9, 0.1]));
  smoother.push(prediction(labels, [0.9, 0.1]));
  smoother.push(null); // nobody in frame
  assert.equal(smoother.current().label, null, 'two frames were not enough to commit');
});
