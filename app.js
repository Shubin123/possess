/*
 * poser - the page.
 *
 * Wires the camera to the estimator, the estimator to the classifier and the
 * classifier to the screen. Everything it touches is local: getUserMedia for
 * the frames, IndexedDB for the pose weights, localStorage for the samples and
 * the trained model.
 */
(function (global) {
  const PZ = global.PZ;
  const doc = global.document;
  const $ = function (id) { return doc.getElementById(id); };

  const MODEL_KEY = 'poser.model.v1';
  const MIN_SAMPLES_PER_CLASS = 10;

  const el = {
    startCamera: $('startCamera'), stopCamera: $('stopCamera'),
    videoFile: $('videoFile'), imageFile: $('imageFile'),
    stage: $('stage'), video: $('video'), stillImage: $('stillImage'),
    overlay: $('overlay'), stageEmpty: $('stageEmpty'),
    mirrorToggle: $('mirrorToggle'), perf: $('perf'),
    modelId: $('modelId'), modelNote: $('modelNote'),
    cacheStatus: $('cacheStatus'), downloadBtn: $('downloadBtn'),
    progressBar: $('progressBar'), progressFill: $('progressFill'),
    status: $('status'),
    backendMode: $('backendMode'), topLabel: $('topLabel'), topScore: $('topScore'),
    scoreList: $('scoreList'), repCounts: $('repCounts'), resetCounts: $('resetCounts'),
    measurements: $('measurements'),
    className: $('className'), recordBtn: $('recordBtn'), recordSeconds: $('recordSeconds'),
    mirrorAugment: $('mirrorAugment'), classList: $('classList'),
    trainBtn: $('trainBtn'), clearData: $('clearData'), trainStatus: $('trainStatus'),
    exportData: $('exportData'), importData: $('importData'),
    exportModel: $('exportModel'), importModel: $('importModel')
  };

  const state = {
    detector: null,
    loading: false,
    stream: null,
    source: null,          // the <video> or <img> currently being read
    running: false,
    lastTimestamp: 0,
    dataset: PZ.dataset.PoseDataset.load(),
    model: null,           // trained MlpClassifier, when there is one
    knn: null,             // rebuilt from the dataset whenever it changes
    smoother: new PZ.classifier.PredictionSmoother(),
    recording: null,
    objectUrl: null,
    frames: 0,
    fpsAt: 0,
    fps: 0,
    latency: 0,
    renderedLabels: null
  };

  const ctx = el.overlay.getContext('2d');

  /* ---------- status helpers ---------- */

  function setStatus(message, isError) {
    el.status.textContent = message || '';
    el.status.classList.toggle('error', !!isError);
  }

  function setTrainStatus(message, isError) {
    el.trainStatus.textContent = message || '';
    el.trainStatus.classList.toggle('error', !!isError);
  }

  function formatBytes(bytes) {
    if (!bytes) return '0 MB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  }

  /* ---------- pose model selection and caching ---------- */

  function currentSpec() { return PZ.pose.MODELS[el.modelId.value] || PZ.pose.MODELS.lite; }

  function populateModels() {
    Object.keys(PZ.pose.MODELS).forEach(function (key) {
      const spec = PZ.pose.MODELS[key];
      const option = doc.createElement('option');
      option.value = key;
      option.textContent = spec.label + ' (' + formatBytes(spec.bytes) + ')';
      el.modelId.appendChild(option);
    });
    el.modelId.value = 'lite';
    el.modelNote.textContent = currentSpec().note;
  }

  async function refreshCacheStatus() {
    const spec = currentSpec();
    const size = await PZ.modelCache.cachedSize(spec.url);
    const cached = size !== null;
    el.cacheStatus.textContent = cached
      ? 'Stored in this browser (' + formatBytes(size) + ')'
      : 'Not downloaded yet (' + formatBytes(spec.bytes) + ')';
    el.cacheStatus.classList.toggle('cached', cached);
    el.downloadBtn.disabled = cached || state.loading;
    el.downloadBtn.textContent = cached ? 'Already downloaded' : 'Download for offline use';
  }

  function onProgress(progress) {
    if (progress.cached) return;
    el.progressBar.classList.remove('hidden');
    const total = progress.total || currentSpec().bytes;
    const pct = total ? Math.min(100, Math.round((progress.loaded / total) * 100)) : 0;
    el.progressFill.style.width = pct + '%';
    el.progressFill.textContent = pct + '%';
    if (progress.done) global.setTimeout(function () { el.progressBar.classList.add('hidden'); }, 600);
  }

  /*
   * Loads the estimator on demand. Every entry point (camera, video, image,
   * the download button) funnels through here so the weights are fetched once.
   */
  async function ensureDetector() {
    if (state.detector && state.detector.spec.id === currentSpec().id) return state.detector;
    if (state.loading) throw new Error('The pose model is still loading');
    state.loading = true;
    el.downloadBtn.disabled = true;
    setStatus('Loading the pose model…');
    try {
      if (state.detector) { state.detector.close(); state.detector = null; }
      state.detector = await PZ.pose.createDetector({ model: el.modelId.value, onProgress: onProgress });
      setStatus('Pose model ready (' + state.detector.delegate + ').');
      return state.detector;
    } catch (err) {
      setStatus('Could not load the pose model: ' + err.message, true);
      throw err;
    } finally {
      state.loading = false;
      refreshCacheStatus();
    }
  }

  /* ---------- sources ---------- */

  function showStage(kind) {
    el.stageEmpty.hidden = kind !== 'none';
    el.video.hidden = kind !== 'video' && kind !== 'camera';
    el.stillImage.hidden = kind !== 'image';
    // Only the camera is mirrored; a video or photo is shown as it was shot.
    el.stage.classList.toggle('mirrored', kind === 'camera' && el.mirrorToggle.checked);
  }

  async function startCamera() {
    try {
      await ensureDetector();
      state.stream = await global.navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false
      });
      // A video file may have left a src behind, which would shadow the stream.
      el.video.removeAttribute('src');
      el.video.srcObject = state.stream;
      el.video.loop = false;
      await el.video.play();
      state.source = el.video;
      showStage('camera');
      el.startCamera.disabled = true;
      el.stopCamera.disabled = false;
      el.recordBtn.disabled = false;
      setStatus('Camera running. Nothing is uploaded.');
      startLoop();
    } catch (err) {
      setStatus(err.name === 'NotAllowedError'
        ? 'Camera permission was denied.'
        : 'Could not start the camera: ' + err.message, true);
    }
  }

  function stopCamera() {
    if (state.stream) {
      state.stream.getTracks().forEach(function (track) { track.stop(); });
      state.stream = null;
    }
    el.video.srcObject = null;
    state.running = false;
    state.source = null;
    stopRecording();
    el.startCamera.disabled = false;
    el.stopCamera.disabled = true;
    el.recordBtn.disabled = true;
    showStage('none');
    ctx.clearRect(0, 0, el.overlay.width, el.overlay.height);
    setStatus('Camera stopped.');
  }

  async function useVideoFile(file) {
    try {
      await ensureDetector();
      stopCamera();
      el.video.srcObject = null;
      setObjectUrl(el.video, file);
      el.video.loop = true;
      await el.video.play();
      state.source = el.video;
      showStage('video');
      el.recordBtn.disabled = false;
      setStatus('Playing ' + file.name + '.');
      startLoop();
    } catch (err) {
      setStatus('Could not play that video: ' + err.message, true);
    }
  }

  async function useImageFile(file) {
    try {
      await ensureDetector();
      stopCamera();
      state.running = false;
      await new Promise(function (resolve, reject) {
        el.stillImage.onload = resolve;
        el.stillImage.onerror = function () { reject(new Error('unsupported image')); };
        setObjectUrl(el.stillImage, file);
      });
      state.source = el.stillImage;
      showStage('image');
      setStatus('Classifying ' + file.name + '.');
      await step(true);
    } catch (err) {
      setStatus('Could not read that image: ' + err.message, true);
    }
  }

  /** Swaps in a new object URL for `element`, releasing the previous one. */
  function setObjectUrl(element, file) {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = URL.createObjectURL(file);
    element.src = state.objectUrl;
    return state.objectUrl;
  }

  function sourceSize(source) {
    return {
      width: source.videoWidth || source.naturalWidth || source.width || 0,
      height: source.videoHeight || source.naturalHeight || source.height || 0
    };
  }

  /* ---------- the frame loop ---------- */

  function startLoop() {
    if (state.running) return;
    state.running = true;
    state.fpsAt = global.performance.now();
    state.frames = 0;
    global.requestAnimationFrame(function loop() {
      if (!state.running) return;
      step(false).then(function () { global.requestAnimationFrame(loop); });
    });
  }

  async function step(isStill) {
    const source = state.source;
    if (!source || !state.detector) return;
    const size = sourceSize(source);
    if (!size.width || !size.height) return;

    if (el.overlay.width !== size.width || el.overlay.height !== size.height) {
      el.overlay.width = size.width;
      el.overlay.height = size.height;
    }

    let result;
    try {
      // detectForVideo rejects a timestamp that is not strictly increasing.
      const timestamp = isStill ? null : Math.max(state.lastTimestamp + 1, Math.round(global.performance.now()));
      if (!isStill) state.lastTimestamp = timestamp;
      result = await state.detector.detect(source, timestamp);
    } catch (err) {
      state.running = false;
      setStatus('Pose estimation stopped: ' + err.message, true);
      return;
    }

    state.latency = result.latencyMs;
    if (isStill) {
      // One frame, and the first one also pays for warming up the graph.
      el.perf.textContent = 'single frame · ' + state.latency.toFixed(0) + ' ms';
    } else {
      state.frames++;
      const now = global.performance.now();
      if (now - state.fpsAt > 500) {
        state.fps = (state.frames * 1000) / (now - state.fpsAt);
        state.frames = 0;
        state.fpsAt = now;
      }
      el.perf.textContent = state.fps.toFixed(0) + ' fps · ' + state.latency.toFixed(0) + ' ms/frame';
    }

    drawPose(result.landmarks, size);

    const feature = result.landmarks
      ? PZ.features.featurize(result.landmarks, { aspect: size.width / size.height })
      : null;
    handleFeature(feature, result.landmarks);
  }

  function drawPose(landmarks, size) {
    ctx.clearRect(0, 0, size.width, size.height);
    if (!landmarks) return;
    const scale = Math.max(1, Math.min(size.width, size.height) / 240);

    ctx.lineWidth = 2.4 * scale;
    ctx.strokeStyle = 'rgba(157, 180, 240, 0.9)';
    ctx.beginPath();
    PZ.features.SKELETON.forEach(function (bone) {
      const a = landmarks[bone[0]], b = landmarks[bone[1]];
      if (!a || !b) return;
      ctx.moveTo(a.x * size.width, a.y * size.height);
      ctx.lineTo(b.x * size.width, b.y * size.height);
    });
    ctx.stroke();

    ctx.fillStyle = '#f6f7fd';
    PZ.features.POINTS.forEach(function (index) {
      const p = landmarks[index];
      if (!p) return;
      ctx.beginPath();
      ctx.arc(p.x * size.width, p.y * size.height, 2.4 * scale, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  /* ---------- classification ---------- */

  function sameLabels(a, b) {
    return a && b && a.length === b.length && a.every(function (label, i) { return label === b[i]; });
  }

  /*
   * Which head answers in "My trained model" mode. A network trained before
   * the newest pose was recorded cannot name it, so while it is stale the
   * nearest-neighbour head - which always covers everything recorded - takes
   * over until the user retrains.
   */
  function activeBackend() {
    if (el.backendMode.value !== 'trained') return null; // the rules answer
    if (state.model && state.knn && !sameLabels(state.model.labels, state.knn.labels)) return state.knn;
    return state.model || state.knn;
  }

  function handleFeature(feature, landmarks) {
    if (!feature) {
      el.topLabel.textContent = '—';
      el.topScore.textContent = 'no pose detected';
      state.smoother.push(null);
      return;
    }

    let prediction = null;
    if (el.backendMode.value === 'trained') {
      const backend = activeBackend();
      prediction = backend ? backend.predict(feature.values) : null;
      if (!prediction) {
        el.topLabel.textContent = '—';
        el.topScore.textContent = 'record a couple of poses first';
      }
    } else {
      prediction = PZ.rules.predict(feature);
      renderMeasurements(prediction.measurements);
    }

    if (prediction) {
      const smoothed = state.smoother.push(feature.visible ? prediction : null);
      const shown = feature.visible ? (smoothed.label || prediction.label) : 'out of frame';
      el.topLabel.textContent = shown;
      el.topScore.textContent = feature.visible
        ? Math.round(smoothed.score * 100) + '% confident'
        : 'step back so your whole body is visible';
      renderScores(prediction.labels, smoothed.scores || prediction.scores, shown);
      renderCounts(smoothed.counts);
    }

    if (state.recording) captureSample(feature, landmarks);
  }

  function renderScores(labels, scores, leader) {
    const key = labels.join('\u0000');
    if (state.renderedLabels !== key) {
      el.scoreList.textContent = '';
      labels.forEach(function (label) {
        const li = doc.createElement('li');
        li.dataset.label = label;
        const name = doc.createElement('span');
        name.textContent = label;
        const bar = doc.createElement('span');
        bar.className = 'bar';
        bar.appendChild(doc.createElement('span'));
        const value = doc.createElement('span');
        value.className = 'value';
        li.append(name, bar, value);
        el.scoreList.appendChild(li);
      });
      state.renderedLabels = key;
    }
    Array.prototype.forEach.call(el.scoreList.children, function (li, i) {
      const score = scores[i] || 0;
      li.querySelector('.bar span').style.width = Math.round(score * 100) + '%';
      li.querySelector('.value').textContent = score.toFixed(2);
      li.classList.toggle('leader', li.dataset.label === leader);
    });
  }

  function renderCounts(counts) {
    const names = Object.keys(counts || {});
    el.repCounts.textContent = '';
    if (!names.length) {
      const li = doc.createElement('li');
      li.className = 'empty';
      li.textContent = 'No pose has been held yet.';
      el.repCounts.appendChild(li);
      return;
    }
    names.forEach(function (name) {
      const li = doc.createElement('li');
      const label = doc.createElement('span');
      label.textContent = name;
      const value = doc.createElement('span');
      value.className = 'count';
      value.textContent = counts[name] + '×';
      li.append(label, value);
      el.repCounts.appendChild(li);
    });
  }

  function renderMeasurements(m) {
    if (!m) return;
    const rows = [
      ['knee angle L/R', Math.round(m.kneeL) + '° / ' + Math.round(m.kneeR) + '°'],
      ['elbow angle L/R', Math.round(m.elbowL) + '° / ' + Math.round(m.elbowR) + '°'],
      ['arm from vertical L/R', Math.round(m.armL) + '° / ' + Math.round(m.armR) + '°'],
      ['torso lean', Math.round(m.torsoLean) + '°'],
      ['hips above knees', m.hipAboveKnee.toFixed(2) + ' torsos'],
      ['shoulder to ankle', m.bodyExtension.toFixed(2) + ' torsos'],
      ['stance width', m.stanceWidth.toFixed(2) + ' torsos']
    ];
    el.measurements.textContent = '';
    rows.forEach(function (row) {
      const li = doc.createElement('li');
      const name = doc.createElement('span');
      name.textContent = row[0];
      const value = doc.createElement('span');
      value.textContent = row[1];
      li.append(name, value);
      el.measurements.appendChild(li);
    });
  }

  /* ---------- recording and training ---------- */

  function captureSample(feature, landmarks) {
    if (!feature.visible) return;
    const label = state.recording.label;
    let added = state.dataset.addSample(label, feature.values);
    if (added && el.mirrorAugment.checked && landmarks) {
      const mirrored = PZ.features.featurize(PZ.features.mirrorLandmarks(landmarks), {
        aspect: el.overlay.width / el.overlay.height
      });
      if (mirrored) state.dataset.addSample(label, mirrored.values, { mirrored: true });
    }
    state.recording.captured++;
    if (!added) {
      state.recording.full = true;
      stopRecording();
      return;
    }
    if (global.performance.now() >= state.recording.until) stopRecording();
    else updateCount(label, state.dataset.counts()[label]);
  }

  /** Refreshes one row's sample count without rebuilding the list. */
  function updateCount(label, count) {
    const row = el.classList.querySelector('[data-label="' + CSS.escape(label) + '"] .count');
    if (row) row.textContent = count + ' samples';
  }

  function startRecording() {
    const label = (el.className.value || '').trim();
    if (!label) { setTrainStatus('Give the pose a name first.', true); return; }
    if (!state.source) { setTrainStatus('Start the camera or open a video first.', true); return; }
    const seconds = Math.min(30, Math.max(1, Number(el.recordSeconds.value) || 4));
    state.dataset.addClass(label);
    state.recording = { label: label, until: global.performance.now() + seconds * 1000, captured: 0 };
    el.recordBtn.classList.add('recording');
    el.recordBtn.textContent = 'Recording…';
    setTrainStatus('Hold "' + label + '" for ' + seconds + ' seconds.');
  }

  function stopRecording() {
    if (!state.recording) return;
    const done = state.recording;
    state.recording = null;
    el.recordBtn.classList.remove('recording');
    el.recordBtn.textContent = 'Record';
    persistDataset();
    rebuildKnn();
    renderClasses();
    const stale = state.model && state.knn && !sameLabels(state.model.labels, state.knn.labels);
    setTrainStatus('Recorded ' + done.captured + ' frames of "' + done.label + '".'
      + (done.full ? ' That pose is now as full as it can get.' : '')
      + (stale ? ' Train again to add it to the model.' : ''));
  }

  function persistDataset() {
    if (!state.dataset.save()) {
      setTrainStatus('Samples could not be saved to this browser (storage is full or blocked).', true);
    }
  }

  /*
   * The nearest-neighbour head is rebuilt after every take, so "My trained
   * model" answers immediately - training the network is then an upgrade
   * rather than a prerequisite.
   */
  function rebuildKnn() {
    const data = state.dataset.toTraining(MIN_SAMPLES_PER_CLASS);
    if (data.labels.length < 2) { state.knn = null; return; }
    state.knn = new PZ.classifier.KnnClassifier({ k: 5 }).fit(data.rows, data.targets, data.labels);
    state.smoother.reset();
  }

  function renderClasses() {
    const counts = state.dataset.counts();
    const labels = state.dataset.labels;
    el.classList.textContent = '';
    if (!labels.length) {
      const li = doc.createElement('li');
      li.className = 'empty';
      li.textContent = 'No poses recorded yet.';
      el.classList.appendChild(li);
    }
    labels.forEach(function (label) {
      const li = doc.createElement('li');
      li.dataset.label = label;
      const name = doc.createElement('span');
      name.textContent = label;
      const count = doc.createElement('span');
      count.className = 'count';
      count.textContent = counts[label] + ' samples';
      const remove = doc.createElement('button');
      remove.type = 'button';
      remove.className = 'link-button';
      remove.textContent = 'Remove';
      remove.addEventListener('click', function () {
        state.dataset.removeClass(label);
        persistDataset();
        rebuildKnn();
        renderClasses();
      });
      li.append(name, count, remove);
      el.classList.appendChild(li);
    });
    el.trainBtn.disabled = state.dataset.trainableLabels(MIN_SAMPLES_PER_CLASS).length < 2;
  }

  async function train() {
    const data = state.dataset.toTraining(MIN_SAMPLES_PER_CLASS);
    if (data.labels.length < 2) {
      setTrainStatus('Record at least ' + MIN_SAMPLES_PER_CLASS + ' samples of two different poses.', true);
      return;
    }
    el.trainBtn.disabled = true;
    setTrainStatus('Training on ' + data.rows.length + ' samples…');
    // Yield once so the status paints before the main thread is busy.
    await new Promise(function (resolve) { global.setTimeout(resolve, 0); });

    try {
      const model = new PZ.classifier.MlpClassifier({ hidden: 32 });
      const started = global.performance.now();
      const report = model.fit(data.rows, data.targets, data.labels, { epochs: 150, seed: 1337 });
      state.model = model;
      state.smoother.reset();
      el.backendMode.value = 'trained';
      saveModel();
      const took = ((global.performance.now() - started) / 1000).toFixed(1);
      const val = report.valAccuracy === null ? 'n/a' : Math.round(report.valAccuracy * 100) + '%';
      setTrainStatus('Trained ' + data.labels.length + ' poses in ' + took + ' s · '
        + Math.round(report.trainAccuracy * 100) + '% on training frames, ' + val + ' on held-out frames.'
        + (data.skipped.length ? ' Skipped ' + data.skipped.join(', ') + ' (too few samples).' : ''));
    } catch (err) {
      setTrainStatus('Training failed: ' + err.message, true);
    } finally {
      el.trainBtn.disabled = false;
    }
  }

  function saveModel() {
    if (!state.model) return;
    try { global.localStorage.setItem(MODEL_KEY, JSON.stringify(state.model.toJSON())); }
    catch (err) { /* the dataset alone can rebuild it */ }
  }

  function loadSavedModel() {
    try {
      const raw = global.localStorage.getItem(MODEL_KEY);
      if (!raw) return;
      state.model = PZ.classifier.MlpClassifier.fromJSON(JSON.parse(raw));
    } catch (err) {
      state.model = null;
    }
  }

  /* ---------- import and export ---------- */

  function download(name, json) {
    const blob = new global.Blob([JSON.stringify(json)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = doc.createElement('a');
    link.href = url;
    link.download = name;
    link.click();
    global.setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function readJson(file) {
    return new Promise(function (resolve, reject) {
      const reader = new global.FileReader();
      reader.onload = function () {
        try { resolve(JSON.parse(reader.result)); }
        catch (err) { reject(new Error('that file is not JSON')); }
      };
      reader.onerror = function () { reject(new Error('could not read that file')); };
      reader.readAsText(file);
    });
  }

  /* ---------- events ---------- */

  el.startCamera.addEventListener('click', startCamera);
  el.stopCamera.addEventListener('click', stopCamera);
  el.videoFile.addEventListener('change', function (e) { if (e.target.files[0]) useVideoFile(e.target.files[0]); });
  el.imageFile.addEventListener('change', function (e) { if (e.target.files[0]) useImageFile(e.target.files[0]); });
  el.mirrorToggle.addEventListener('change', function () {
    el.stage.classList.toggle('mirrored', el.mirrorToggle.checked && !!state.stream);
  });
  el.modelId.addEventListener('change', function () {
    el.modelNote.textContent = currentSpec().note;
    refreshCacheStatus();
    if (state.detector) setStatus('Restart the camera to switch to the ' + currentSpec().label + '.');
  });
  el.downloadBtn.addEventListener('click', async function () {
    el.downloadBtn.disabled = true;
    try {
      await PZ.modelCache.fetchModel(currentSpec().url, onProgress);
      setStatus(currentSpec().label + ' stored for offline use.');
    } catch (err) {
      setStatus('Download failed: ' + err.message, true);
    }
    refreshCacheStatus();
  });
  el.backendMode.addEventListener('change', function () {
    state.smoother.reset();
    state.renderedLabels = null;
    if (el.backendMode.value === 'trained' && !activeBackend()) {
      setTrainStatus('No model yet - record two poses below, then train.');
    }
  });
  el.resetCounts.addEventListener('click', function () {
    state.smoother.counts = {};
    renderCounts({});
  });
  el.recordBtn.addEventListener('click', function () {
    if (state.recording) stopRecording(); else startRecording();
  });
  el.trainBtn.addEventListener('click', train);
  el.clearData.addEventListener('click', function () {
    state.dataset.clear();
    state.model = null;
    state.knn = null;
    state.smoother.reset();
    state.renderedLabels = null;
    PZ.dataset.PoseDataset.clearStorage();
    try { global.localStorage.removeItem(MODEL_KEY); } catch (err) { /* nothing stored */ }
    renderClasses();
    setTrainStatus('All samples cleared.');
  });
  el.exportData.addEventListener('click', function () { download('poser-samples.json', state.dataset.toJSON()); });
  el.exportModel.addEventListener('click', function () {
    if (!state.model) { setTrainStatus('Train a model before exporting it.', true); return; }
    download('poser-model.json', state.model.toJSON());
  });
  el.importData.addEventListener('change', async function (e) {
    const file = e.target.files[0];
    if (!file) return;
    try {
      state.dataset = PZ.dataset.PoseDataset.fromJSON(await readJson(file));
      persistDataset();
      rebuildKnn();
      renderClasses();
      setTrainStatus('Imported ' + state.dataset.samples.length + ' samples.');
    } catch (err) {
      setTrainStatus('Could not import those samples: ' + err.message, true);
    }
  });
  el.importModel.addEventListener('change', async function (e) {
    const file = e.target.files[0];
    if (!file) return;
    try {
      state.model = PZ.classifier.MlpClassifier.fromJSON(await readJson(file));
      state.smoother.reset();
      state.renderedLabels = null;
      el.backendMode.value = 'trained';
      saveModel();
      setTrainStatus('Imported a model for ' + state.model.labels.join(', ') + '.');
    } catch (err) {
      setTrainStatus('Could not import that model: ' + err.message, true);
    }
  });

  /* ---------- start ---------- */

  populateModels();
  refreshCacheStatus();
  loadSavedModel();
  rebuildKnn();
  renderClasses();
  renderCounts({});
  if (state.model) el.backendMode.value = 'trained';
  showStage('none');

  // Exposed for the end-to-end test, which drives the page without a camera.
  global.__poser = state;
})(typeof globalThis !== 'undefined' ? globalThis : this);
