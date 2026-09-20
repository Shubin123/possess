/*
 * possess - the pose classifier itself.
 *
 * Two heads share one feature vector (features.js, 60 dims):
 *
 *   KNN  - remembers every recorded sample and votes among the nearest ones.
 *          Zero training time, so it answers the moment a class is recorded.
 *   MLP  - one hidden ReLU layer and a softmax output, trained with Adam.
 *          Generalizes better than KNN once there are a few hundred samples,
 *          and serializes to a few kilobytes of JSON.
 *
 * Written in plain JavaScript rather than against a tensor library: the input
 * is 60 numbers wide and the datasets are a few thousand rows, so training
 * finishes in well under a second on the main thread, the whole thing runs in
 * `node --test`, and the page needs no ML runtime for the classifier at all.
 */
(function (global) {
  const PZ = (global.PZ = global.PZ || {});

  /** Deterministic PRNG, so training runs are reproducible in tests. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function zeros(n) { return new Float32Array(n); }

  /** Per-feature mean/std, so no single dimension dominates the distance. */
  function fitScaler(rows) {
    const d = rows[0].length;
    const mean = new Float32Array(d);
    const std = new Float32Array(d);
    for (let i = 0; i < rows.length; i++) {
      for (let j = 0; j < d; j++) mean[j] += rows[i][j];
    }
    for (let j = 0; j < d; j++) mean[j] /= rows.length;
    for (let i = 0; i < rows.length; i++) {
      for (let j = 0; j < d; j++) {
        const dev = rows[i][j] - mean[j];
        std[j] += dev * dev;
      }
    }
    for (let j = 0; j < d; j++) {
      std[j] = Math.sqrt(std[j] / rows.length);
      if (std[j] < 1e-6) std[j] = 1;
    }
    return { mean: mean, std: std };
  }

  function applyScaler(scaler, row) {
    const out = new Float32Array(row.length);
    for (let j = 0; j < row.length; j++) out[j] = (row[j] - scaler.mean[j]) / scaler.std[j];
    return out;
  }

  function softmax(logits) {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
    const out = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - max); sum += out[i]; }
    for (let i = 0; i < logits.length; i++) out[i] /= sum || 1;
    return out;
  }

  function argmax(values) {
    let best = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
    return best;
  }

  /** Packs scores into the shape the UI renders: sorted, labelled, with a top. */
  function toPrediction(labels, scores) {
    const ranked = [];
    for (let i = 0; i < labels.length; i++) ranked.push({ label: labels[i], score: scores[i] });
    ranked.sort(function (a, b) { return b.score - a.score; });
    return {
      label: ranked[0].label,
      score: ranked[0].score,
      labels: labels.slice(),
      scores: Array.from(scores),
      ranked: ranked
    };
  }

  /*
   * Nearest-neighbour head. Distance-weighted vote over the k closest samples,
   * on standardized features.
   */
  function KnnClassifier(options) {
    const opts = options || {};
    this.k = opts.k || 5;
    this.labels = [];
    this.rows = [];
    this.rawRows = [];
    this.targets = [];
    this.scaler = null;
  }

  KnnClassifier.prototype.fit = function (rows, targets, labels) {
    if (!rows.length) throw new Error('No samples to fit');
    this.labels = labels.slice();
    this.rawRows = rows.slice();
    this.scaler = fitScaler(rows);
    this.rows = rows.map((row) => applyScaler(this.scaler, row));
    this.targets = targets.slice();
    return this;
  };

  KnnClassifier.prototype.predict = function (row) {
    if (!this.rows.length) return null;
    const q = applyScaler(this.scaler, row);
    const k = Math.min(this.k, this.rows.length);
    // Partial selection: the datasets here are small enough that a linear scan
    // with a k-sized insertion list beats sorting every distance.
    const best = [];
    for (let i = 0; i < this.rows.length; i++) {
      let sum = 0;
      const r = this.rows[i];
      for (let j = 0; j < q.length; j++) { const d = q[j] - r[j]; sum += d * d; }
      const dist = Math.sqrt(sum);
      if (best.length < k) {
        best.push({ dist: dist, target: this.targets[i], index: i });
        best.sort(function (a, b) { return a.dist - b.dist; });
      } else if (dist < best[k - 1].dist) {
        best[k - 1] = { dist: dist, target: this.targets[i], index: i };
        best.sort(function (a, b) { return a.dist - b.dist; });
      }
    }
    const scores = zeros(this.labels.length);
    let total = 0;
    for (let i = 0; i < best.length; i++) {
      const w = 1 / (best[i].dist + 1e-3);
      scores[best[i].target] += w;
      total += w;
    }
    for (let i = 0; i < scores.length; i++) scores[i] /= total || 1;
    const pred = toPrediction(this.labels, scores);

    let minRawDist = Infinity;
    if (this.rawRows && best.length) {
      const nearestIdx = best[0].index;
      const nearestRaw = this.rawRows[nearestIdx];
      let sumSq = 0;
      for (let j = 0; j < row.length; j++) {
        const diff = row[j] - nearestRaw[j];
        sumSq += diff * diff;
      }
      minRawDist = Math.sqrt(sumSq);
    }
    pred.minRawDist = minRawDist;
    const matchConfidence = Math.max(0, Math.min(1, Math.exp(-Math.pow(minRawDist / 0.85, 2))));
    pred.matchConfidence = matchConfidence;

    if (this.labels.length === 1) {
      scores[0] = matchConfidence;
      pred.scores = [matchConfidence];
      pred.score = matchConfidence;
      pred.ranked = [{ label: this.labels[0], score: matchConfidence }];
      if (matchConfidence < 0.35) {
        pred.label = 'unknown';
      }
    }
    return pred;
  };

  /*
   * Softmax classifier with one hidden ReLU layer, trained with Adam and L2.
   * Weights are flat Float32Arrays in row-major order.
   */
  function MlpClassifier(options) {
    const opts = options || {};
    this.hidden = opts.hidden || 32;
    this.labels = [];
    this.scaler = null;
    this.w1 = null; this.b1 = null; this.w2 = null; this.b2 = null;
    this.trained = false;
  }

  MlpClassifier.prototype._init = function (d, c, rand) {
    // He initialization for the ReLU layer, Xavier for the linear output.
    const h = this.hidden;
    this.w1 = zeros(d * h); this.b1 = zeros(h);
    this.w2 = zeros(h * c); this.b2 = zeros(c);
    const s1 = Math.sqrt(2 / d), s2 = Math.sqrt(1 / h);
    for (let i = 0; i < this.w1.length; i++) this.w1[i] = (rand() * 2 - 1) * s1;
    for (let i = 0; i < this.w2.length; i++) this.w2[i] = (rand() * 2 - 1) * s2;
  };

  MlpClassifier.prototype._forward = function (x) {
    const d = x.length, h = this.hidden, c = this.b2.length;
    const hid = zeros(h);
    for (let j = 0; j < h; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < d; i++) sum += x[i] * this.w1[i * h + j];
      hid[j] = sum > 0 ? sum : 0;
    }
    const logits = zeros(c);
    for (let k = 0; k < c; k++) {
      let sum = this.b2[k];
      for (let j = 0; j < h; j++) sum += hid[j] * this.w2[j * c + k];
      logits[k] = sum;
    }
    return { hidden: hid, probs: softmax(logits) };
  };

  /*
   * `rows` are raw feature vectors, `targets` class indices, `labels` names.
   * Returns { epochs, loss, trainAccuracy, valAccuracy, samples } - the numbers
   * the page shows after training so the user can see whether it worked.
   */
  MlpClassifier.prototype.fit = function (rows, targets, labels, options) {
    const opts = options || {};
    const epochs = opts.epochs || 120;
    const batchSize = opts.batchSize || 16;
    const lr = opts.learningRate || 0.01;
    const l2 = opts.l2 === undefined ? 1e-4 : opts.l2;
    const valSplit = opts.validationSplit === undefined ? 0.2 : opts.validationSplit;
    const rand = mulberry32(opts.seed === undefined ? 1337 : opts.seed);
    const onEpoch = opts.onEpoch;

    if (!rows.length) throw new Error('No samples to train on');
    if (labels.length < 2) throw new Error('Need at least two classes to train');

    this.labels = labels.slice();
    this.scaler = fitScaler(rows);
    const scaled = rows.map((row) => applyScaler(this.scaler, row));

    const split = stratifiedSplit(targets, labels.length, valSplit, rand);
    const trainIdx = split.train, valIdx = split.val;
    if (!trainIdx.length) throw new Error('No samples left to train on after the validation split');

    const d = scaled[0].length, c = labels.length, h = this.hidden;
    this._init(d, c, rand);

    // Adam moments, one pair per parameter tensor.
    const m = { w1: zeros(d * h), b1: zeros(h), w2: zeros(h * c), b2: zeros(c) };
    const v = { w1: zeros(d * h), b1: zeros(h), w2: zeros(h * c), b2: zeros(c) };
    const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    let step = 0;
    let lastLoss = 0;

    const gw1 = zeros(d * h), gb1 = zeros(h), gw2 = zeros(h * c), gb2 = zeros(c);

    for (let epoch = 0; epoch < epochs; epoch++) {
      shuffle(trainIdx, rand);
      let epochLoss = 0;

      for (let start = 0; start < trainIdx.length; start += batchSize) {
        const batch = trainIdx.slice(start, start + batchSize);
        gw1.fill(0); gb1.fill(0); gw2.fill(0); gb2.fill(0);

        for (let bi = 0; bi < batch.length; bi++) {
          const x = scaled[batch[bi]];
          const y = targets[batch[bi]];
          const fwd = this._forward(x);
          const probs = fwd.probs, hid = fwd.hidden;
          epochLoss += -Math.log(Math.max(probs[y], 1e-9));

          // dL/dlogits for softmax + cross-entropy.
          const dLogits = zeros(c);
          for (let k = 0; k < c; k++) dLogits[k] = probs[k] - (k === y ? 1 : 0);

          const dHidden = zeros(h);
          for (let j = 0; j < h; j++) {
            let acc = 0;
            for (let k = 0; k < c; k++) {
              gw2[j * c + k] += hid[j] * dLogits[k];
              acc += this.w2[j * c + k] * dLogits[k];
            }
            dHidden[j] = hid[j] > 0 ? acc : 0; // ReLU gate
          }
          for (let k = 0; k < c; k++) gb2[k] += dLogits[k];
          for (let j = 0; j < h; j++) {
            if (dHidden[j] === 0) continue;
            gb1[j] += dHidden[j];
            for (let i = 0; i < d; i++) gw1[i * h + j] += x[i] * dHidden[j];
          }
        }

        step++;
        const invN = 1 / batch.length;
        const bc1 = 1 - Math.pow(beta1, step);
        const bc2 = 1 - Math.pow(beta2, step);
        const adam = function (params, grads, mo, vo, decay) {
          for (let i = 0; i < params.length; i++) {
            const g = grads[i] * invN + decay * params[i];
            mo[i] = beta1 * mo[i] + (1 - beta1) * g;
            vo[i] = beta2 * vo[i] + (1 - beta2) * g * g;
            params[i] -= lr * (mo[i] / bc1) / (Math.sqrt(vo[i] / bc2) + eps);
          }
        };
        adam(this.w1, gw1, m.w1, v.w1, l2);
        adam(this.b1, gb1, m.b1, v.b1, 0);
        adam(this.w2, gw2, m.w2, v.w2, l2);
        adam(this.b2, gb2, m.b2, v.b2, 0);
      }

      lastLoss = epochLoss / trainIdx.length;
      if (onEpoch) onEpoch(epoch + 1, lastLoss);
    }

    this.trained = true;
    const accuracy = (idx) => {
      if (!idx.length) return null;
      let hits = 0;
      for (let i = 0; i < idx.length; i++) {
        if (argmax(this._forward(scaled[idx[i]]).probs) === targets[idx[i]]) hits++;
      }
      return hits / idx.length;
    };
    return {
      epochs: epochs,
      loss: lastLoss,
      trainAccuracy: accuracy(trainIdx),
      valAccuracy: accuracy(valIdx),
      samples: rows.length,
      validationSamples: valIdx.length
    };
  };

  MlpClassifier.prototype.predict = function (row) {
    if (!this.trained) return null;
    return toPrediction(this.labels, this._forward(applyScaler(this.scaler, row)).probs);
  };

  MlpClassifier.prototype.toJSON = function () {
    if (!this.trained) throw new Error('Model is not trained');
    return {
      format: 'possess-mlp',
      version: 1,
      labels: this.labels.slice(),
      hidden: this.hidden,
      inputSize: this.scaler.mean.length,
      featureNames: PZ.features ? PZ.features.FEATURE_NAMES : undefined,
      scaler: { mean: Array.from(this.scaler.mean), std: Array.from(this.scaler.std) },
      weights: {
        w1: Array.from(this.w1), b1: Array.from(this.b1),
        w2: Array.from(this.w2), b2: Array.from(this.b2)
      }
    };
  };

  MlpClassifier.fromJSON = function (json) {
    if (!json || json.format !== 'possess-mlp') {
      throw new Error('Not a possess model file');
    }
    const model = new MlpClassifier({ hidden: json.hidden });
    model.labels = json.labels.slice();
    model.scaler = {
      mean: Float32Array.from(json.scaler.mean),
      std: Float32Array.from(json.scaler.std)
    };
    model.w1 = Float32Array.from(json.weights.w1);
    model.b1 = Float32Array.from(json.weights.b1);
    model.w2 = Float32Array.from(json.weights.w2);
    model.b2 = Float32Array.from(json.weights.b2);
    if (model.w1.length !== model.scaler.mean.length * model.hidden) {
      throw new Error('Model weights do not match the declared shape');
    }
    model.trained = true;
    return model;
  };

  function shuffle(array, rand) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const t = array[i]; array[i] = array[j]; array[j] = t;
    }
    return array;
  }

  /*
   * Holds out `fraction` of every class, so the validation accuracy means
   * something even when one pose has far more samples than another.
   */
  function stratifiedSplit(targets, classCount, fraction, rand) {
    const byClass = [];
    for (let i = 0; i < classCount; i++) byClass.push([]);
    for (let i = 0; i < targets.length; i++) byClass[targets[i]].push(i);
    const train = [], val = [];
    for (let c = 0; c < classCount; c++) {
      const idx = shuffle(byClass[c].slice(), rand);
      // Never hold out so much that a class disappears from training.
      const take = Math.min(Math.floor(idx.length * fraction), Math.max(0, idx.length - 1));
      for (let i = 0; i < idx.length; i++) (i < take ? val : train).push(idx[i]);
    }
    return { train: train, val: val };
  }

  /*
   * Predictions from a live camera flicker between neighbouring classes. This
   * smooths the probability vector with an EMA, then only switches the reported
   * label once the leader has been ahead for `holdFrames` frames and clears
   * `threshold` - which also gives a clean signal to count repetitions with.
   */
  function PredictionSmoother(options) {
    const opts = options || {};
    this.alpha = opts.alpha === undefined ? 0.35 : opts.alpha;
    this.threshold = opts.threshold === undefined ? 0.6 : opts.threshold;
    this.holdFrames = opts.holdFrames === undefined ? 3 : opts.holdFrames;
    this.reset();
  }

  PredictionSmoother.prototype.reset = function () {
    this.scores = null;
    this.labels = null;
    this.candidate = null;
    this.candidateFrames = 0;
    this.label = null;
    this.score = 0;
    this.counts = {};
  };

  PredictionSmoother.prototype.push = function (prediction) {
    if (!prediction) {
      this.candidate = null;
      this.candidateFrames = 0;
      return this.current();
    }
    // A retrained model can change the class list; start the average over
    // rather than blending scores that mean different things.
    const labels = prediction.labels;
    if (!this.labels || this.labels.join('\u0000') !== labels.join('\u0000')) {
      this.labels = labels.slice();
      this.scores = prediction.scores.slice();
      this.label = null;
      this.candidate = null;
      this.candidateFrames = 0;
    } else {
      for (let i = 0; i < prediction.scores.length; i++) {
        this.scores[i] += this.alpha * (prediction.scores[i] - this.scores[i]);
      }
    }

    let best = 0;
    for (let i = 1; i < this.scores.length; i++) if (this.scores[i] > this.scores[best]) best = i;
    const label = this.labels[best];
    const score = this.scores[best];

    if (label === this.candidate) this.candidateFrames++;
    else { this.candidate = label; this.candidateFrames = 1; }

    // Only commit once the leader has held for a few frames and is confident;
    // each commit to a new label is one repetition of that pose.
    if (score >= this.threshold && this.candidateFrames >= this.holdFrames && label !== this.label) {
      this.label = label;
      this.counts[label] = (this.counts[label] || 0) + 1;
    }
    this.score = score;
    return this.current();
  };

  PredictionSmoother.prototype.current = function () {
    return {
      label: this.label,
      score: this.score,
      counts: this.counts,
      labels: this.labels ? this.labels.slice() : null,
      scores: this.scores ? Array.from(this.scores) : null
    };
  };

  PZ.classifier = {
    KnnClassifier: KnnClassifier,
    MlpClassifier: MlpClassifier,
    PredictionSmoother: PredictionSmoother,
    fitScaler: fitScaler,
    applyScaler: applyScaler,
    softmax: softmax,
    argmax: argmax,
    stratifiedSplit: stratifiedSplit,
    mulberry32: mulberry32,
    toPrediction: toPrediction
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = PZ.classifier;
})(typeof globalThis !== 'undefined' ? globalThis : this);
