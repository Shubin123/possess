/*
 * The recorded samples: counting, capping, persistence and the JSON files the
 * page exports. A fake storage stands in for localStorage, including the case
 * everyone forgets - the quota being full.
 */
const test = require('node:test');
const assert = require('node:assert/strict');

const features = require('../features.js');
const { PoseDataset } = require('../dataset.js');
const { makeLandmarks } = require('./fixtures.js');

function fakeStorage(options) {
  const opts = options || {};
  const items = {};
  return {
    items: items,
    setItem: function (key, value) {
      if (opts.full) { const err = new Error('QuotaExceededError'); err.name = 'QuotaExceededError'; throw err; }
      items[key] = String(value);
    },
    getItem: function (key) { return Object.prototype.hasOwnProperty.call(items, key) ? items[key] : null; },
    removeItem: function (key) { delete items[key]; }
  };
}

function sampleFor(pose) {
  return features.featurize(makeLandmarks(pose), { aspect: 1 }).values;
}

function fill(dataset, label, pose, count) {
  for (let i = 0; i < count; i++) dataset.addSample(label, sampleFor(pose));
  return dataset;
}

test('counts samples per pose', () => {
  const dataset = new PoseDataset();
  fill(dataset, 'stand', 'standing', 12);
  fill(dataset, 'tee', 't-pose', 5);
  assert.deepEqual(dataset.counts(), { stand: 12, tee: 5 });
  assert.deepEqual(dataset.labels, ['stand', 'tee']);
});

test('rejects a nameless pose', () => {
  const dataset = new PoseDataset();
  assert.throws(() => dataset.addSample('   ', sampleFor('standing')), /name is required/i);
});

test('stops accepting samples once a pose is full', () => {
  const dataset = new PoseDataset({ maxPerClass: 4 });
  for (let i = 0; i < 4; i++) assert.equal(dataset.addSample('stand', sampleFor('standing')), true);
  assert.equal(dataset.addSample('stand', sampleFor('standing')), false, 'the fifth is refused');
  assert.equal(dataset.counts().stand, 4);
});

test('only offers classes with enough samples for training', () => {
  const dataset = new PoseDataset();
  fill(dataset, 'stand', 'standing', 30);
  fill(dataset, 'tee', 't-pose', 30);
  fill(dataset, 'barely', 'squat', 2);
  const training = dataset.toTraining(10);
  assert.deepEqual(training.labels, ['stand', 'tee']);
  assert.deepEqual(training.skipped, ['barely']);
  assert.equal(training.rows.length, 60);
  assert.equal(training.targets.filter(function (t) { return t === 1; }).length, 30);
});

test('removing a pose removes its samples', () => {
  const dataset = new PoseDataset();
  fill(dataset, 'stand', 'standing', 6);
  fill(dataset, 'tee', 't-pose', 6);
  dataset.removeClass('stand');
  assert.deepEqual(dataset.labels, ['tee']);
  assert.equal(dataset.samples.length, 6);
});

test('survives a save and load round trip', () => {
  const storage = fakeStorage();
  const dataset = new PoseDataset();
  fill(dataset, 'stand', 'standing', 8);
  dataset.addSample('stand', sampleFor('standing'), { mirrored: true });
  assert.equal(dataset.save(storage), true);

  const loaded = PoseDataset.load(storage);
  assert.deepEqual(loaded.counts(), { stand: 9 });
  assert.equal(loaded.samples[8].mirrored, true);
  // Stored at four decimals, which is far finer than the classifier resolves.
  const before = dataset.samples[0].values, after = loaded.samples[0].values;
  for (let i = 0; i < before.length; i++) assert.ok(Math.abs(before[i] - after[i]) < 1e-3);
});

test('reports rather than throws when storage is full', () => {
  const dataset = fill(new PoseDataset(), 'stand', 'standing', 3);
  assert.equal(dataset.save(fakeStorage({ full: true })), false);
});

test('starts empty when storage holds nothing usable', () => {
  const storage = fakeStorage();
  assert.equal(PoseDataset.load(storage).samples.length, 0);
  storage.setItem('poser.dataset.v1', 'not json at all');
  assert.equal(PoseDataset.load(storage).samples.length, 0, 'corrupt storage is ignored');
});

test('clearing removes both the samples and what was stored', () => {
  const storage = fakeStorage();
  const dataset = fill(new PoseDataset(), 'stand', 'standing', 4);
  dataset.save(storage);
  dataset.clear();
  PoseDataset.clearStorage(storage);
  assert.equal(dataset.samples.length, 0);
  assert.equal(PoseDataset.load(storage).samples.length, 0);
});

test('refuses an import that is not a poser dataset', () => {
  assert.throws(() => PoseDataset.fromJSON({ format: 'nope' }), /not a poser dataset/i);
  assert.throws(
    () => PoseDataset.fromJSON({ format: 'poser-dataset', featureLength: 7, samples: [] }),
    /different feature layout/i
  );
});

test('imports a dataset exported by the page', () => {
  const dataset = fill(new PoseDataset(), 'stand', 'standing', 5);
  fill(dataset, 'tee', 't-pose', 5);
  const imported = PoseDataset.fromJSON(JSON.parse(JSON.stringify(dataset.toJSON())));
  assert.deepEqual(imported.counts(), { stand: 5, tee: 5 });
  assert.equal(imported.toTraining(5).labels.length, 2);
});
