/*
 * possess - the recorded training samples.
 *
 * Holds one feature vector per captured frame, grouped by pose name, and keeps
 * the whole thing in localStorage so a session survives a reload. Samples are
 * 60 floats each; rounding to four decimals and capping each class keeps a
 * full dataset comfortably inside the storage quota (~250 KB for 1,000
 * samples) while staying far below the precision the classifier cares about.
 *
 * Nothing here touches the network: export and import move a JSON file through
 * the browser's own download and file-picker.
 */
(function (global) {
  const PZ = (global.PZ = global.PZ || {});

  const STORAGE_KEY = 'possess.dataset.v1';
  const DEFAULT_MAX_PER_CLASS = 600;

  function round4(value) { return Math.round(value * 1e4) / 1e4; }

  function PoseDataset(options) {
    const opts = options || {};
    this.maxPerClass = opts.maxPerClass || DEFAULT_MAX_PER_CLASS;
    this.labels = [];
    this.samples = [];
  }

  PoseDataset.prototype.addClass = function (label) {
    const name = String(label || '').trim();
    if (!name) throw new Error('A pose name is required');
    if (this.labels.indexOf(name) === -1) this.labels.push(name);
    return name;
  };

  PoseDataset.prototype.counts = function () {
    const counts = {};
    this.labels.forEach(function (label) { counts[label] = 0; });
    this.samples.forEach(function (s) { counts[s.label] = (counts[s.label] || 0) + 1; });
    return counts;
  };

  /*
   * Records one frame. Returns false when the class is already full, so the UI
   * can stop a capture run instead of silently dropping frames.
   */
  PoseDataset.prototype.addSample = function (label, values, meta) {
    this.addClass(label);
    const counts = this.counts();
    if (counts[label] >= this.maxPerClass) return false;
    this.samples.push({
      label: label,
      values: Float32Array.from(values),
      mirrored: !!(meta && meta.mirrored)
    });
    return true;
  };

  PoseDataset.prototype.removeClass = function (label) {
    this.labels = this.labels.filter(function (l) { return l !== label; });
    this.samples = this.samples.filter(function (s) { return s.label !== label; });
  };

  PoseDataset.prototype.clear = function () {
    this.labels = [];
    this.samples = [];
  };

  /** Classes with at least `min` samples - the ones worth training on. */
  PoseDataset.prototype.trainableLabels = function (min) {
    const counts = this.counts();
    const floor = min === undefined ? 5 : min;
    return this.labels.filter(function (label) { return counts[label] >= floor; });
  };

  /*
   * Materializes the training matrices. Classes that are too small are left
   * out entirely rather than training a class the model cannot learn; the
   * caller reports which ones were skipped.
   */
  PoseDataset.prototype.toTraining = function (minPerClass) {
    const labels = this.trainableLabels(minPerClass);
    const index = {};
    labels.forEach(function (label, i) { index[label] = i; });
    const rows = [], targets = [];
    this.samples.forEach(function (s) {
      if (index[s.label] === undefined) return;
      rows.push(s.values);
      targets.push(index[s.label]);
    });
    const skipped = this.labels.filter(function (l) { return labels.indexOf(l) === -1; });
    return { rows: rows, targets: targets, labels: labels, skipped: skipped };
  };

  PoseDataset.prototype.toJSON = function () {
    return {
      format: 'possess-dataset',
      version: 1,
      featureLength: PZ.features ? PZ.features.FEATURE_LENGTH : undefined,
      featureNames: PZ.features ? PZ.features.FEATURE_NAMES : undefined,
      labels: this.labels.slice(),
      samples: this.samples.map(function (s) {
        return { label: s.label, mirrored: s.mirrored, values: Array.from(s.values).map(round4) };
      })
    };
  };

  PoseDataset.fromJSON = function (json, options) {
    if (!json || json.format !== 'possess-dataset') {
      throw new Error('Not a possess dataset file');
    }
    const expected = PZ.features ? PZ.features.FEATURE_LENGTH : null;
    if (expected && json.featureLength && json.featureLength !== expected) {
      throw new Error('Dataset was recorded with a different feature layout');
    }
    const dataset = new PoseDataset(options);
    dataset.labels = (json.labels || []).slice();
    (json.samples || []).forEach(function (s) {
      dataset.samples.push({ label: s.label, values: Float32Array.from(s.values), mirrored: !!s.mirrored });
    });
    dataset.labels.forEach(function (l) { dataset.addClass(l); });
    return dataset;
  };

  /** Persists to localStorage. Returns false if storage is full or blocked. */
  PoseDataset.prototype.save = function (storage) {
    const store = storage || safeStorage();
    if (!store) return false;
    try {
      store.setItem(STORAGE_KEY, JSON.stringify(this.toJSON()));
      return true;
    } catch (err) {
      return false;
    }
  };

  PoseDataset.load = function (storage, options) {
    const store = storage || safeStorage();
    if (!store) return new PoseDataset(options);
    try {
      const raw = store.getItem(STORAGE_KEY);
      if (!raw) return new PoseDataset(options);
      return PoseDataset.fromJSON(JSON.parse(raw), options);
    } catch (err) {
      return new PoseDataset(options);
    }
  };

  PoseDataset.clearStorage = function (storage) {
    const store = storage || safeStorage();
    if (store) {
      try {
        store.removeItem(STORAGE_KEY);
      } catch (err) { /* ignore */ }
    }
  };

  function safeStorage() {
    try { return global.localStorage || null; } catch (err) { return null; }
  }

  PZ.dataset = { PoseDataset: PoseDataset, STORAGE_KEY: STORAGE_KEY };

  if (typeof module !== 'undefined' && module.exports) module.exports = PZ.dataset;
})(typeof globalThis !== 'undefined' ? globalThis : this);
