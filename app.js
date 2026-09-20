/*
 * possess - the page.
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

  const MODEL_KEY = 'possess.model.v1';
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
    exportModel: $('exportModel'), importModel: $('importModel'),

    // Health suite elements
    postureReadout: $('postureReadout'), postureScoreNum: $('postureScoreNum'),
    postureScoreBadge: $('postureScoreBadge'), postureLabel: $('postureLabel'),
    postureAdvice: $('postureAdvice'), postureHeadVal: $('postureHeadVal'),
    postureSpineVal: $('postureSpineVal'), postureShoulderVal: $('postureShoulderVal'),
    calibratePosture: $('calibratePosture'), postureAlertToggle: $('postureAlertToggle'),

    routineSelect: $('routineSelect'), routineStepBadge: $('routineStepBadge'),
    routineRepBadge: $('routineRepBadge'), routineStepName: $('routineStepName'),
    routineCue: $('routineCue'), routineHoldTimer: $('routineHoldTimer'),
    routineProgressBar: $('routineProgressBar'), routineToggleBtn: $('routineToggleBtn'),
    routineResetBtn: $('routineResetBtn'),

    romJointSelect: $('romJointSelect'), romCurrentDeg: $('romCurrentDeg'),
    romPeakDeg: $('romPeakDeg'), romNormLabel: $('romNormLabel'),
    romProgressFill: $('romProgressFill'), romTestBtn: $('romTestBtn'),
    romTestStatus: $('romTestStatus'), exportHealthReport: $('exportHealthReport')
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
    renderedLabels: null,
    health: (PZ.health ? {
      posture: new PZ.health.PostureMonitor(),
      routine: new PZ.health.RoutineEngine(true),
      rom: new PZ.health.RomGoniometer(),
      session: new PZ.health.HealthSessionManager(),
      lastLandmarks: null
    } : null)
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

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Outer subtle luminescent halo
    ctx.lineWidth = 3.6 * scale;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
    ctx.beginPath();
    PZ.features.SKELETON.forEach(function (bone) {
      const a = landmarks[bone[0]], b = landmarks[bone[1]];
      if (!a || !b) return;
      ctx.moveTo(a.x * size.width, a.y * size.height);
      ctx.lineTo(b.x * size.width, b.y * size.height);
    });
    ctx.stroke();

    // Inner crisp beam
    ctx.lineWidth = 1.8 * scale;
    ctx.strokeStyle = 'rgba(224, 242, 254, 0.95)';
    ctx.stroke();

    // Joints: two-tier glowing nodes
    PZ.features.POINTS.forEach(function (index) {
      const p = landmarks[index];
      if (!p) return;
      const x = p.x * size.width, y = p.y * size.height;

      // Outer soft halo
      ctx.beginPath();
      ctx.arc(x, y, 4.2 * scale, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(56, 189, 248, 0.35)';
      ctx.fill();

      // Inner crisp node
      ctx.beginPath();
      ctx.arc(x, y, 2.0 * scale, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    });
    ctx.restore();
  }

  /* ---------- classification ---------- */

  function sameLabels(a, b) {
    return a && b && a.length === b.length && a.every(function (label, i) { return label === b[i]; });
  }

  /*
   * Which head answers for custom poses. A network trained before the newest
   * pose was recorded cannot name it, so while it is stale the nearest-neighbour
   * head - which always covers everything recorded - takes over until the user
   * retrains.
   */
  function activeBackend() {
    if (state.model && state.knn && !sameLabels(state.model.labels, state.knn.labels)) return state.knn;
    return state.model || state.knn;
  }

  function combinePredictions(rulePred, customPred) {
    if (!customPred || !customPred.labels.length) return rulePred;
    if (!rulePred || !rulePred.labels.length) return customPred;

    const customConfidence = customPred.matchConfidence !== undefined
      ? customPred.matchConfidence
      : customPred.score;
    const rulePeak = rulePred.peak !== undefined ? rulePred.peak : rulePred.score;

    const labels = [];
    const scores = [];
    const ranked = [];
    const seen = new Set();

    // 1. Custom poses first (stable ordering, slight priority boost for user-taught poses)
    for (let i = 0; i < customPred.labels.length; i++) {
      const lbl = customPred.labels[i];
      seen.add(lbl);
      const baseScore = customPred.labels.length === 1
        ? (customPred.scores[i] || 0)
        : (customPred.scores[i] || 0) * customConfidence;
      // When a user explicitly teaches a custom pose, give it a 15% priority boost over generic rules
      const rawScore = baseScore >= 0.35 ? Math.min(1, baseScore * 1.15) : baseScore;
      labels.push(lbl);
      scores.push(rawScore);
      ranked.push({ label: lbl, score: rawScore, isCustom: true });
    }

    // 2. Built-in rule poses (stable ordering)
    for (let j = 0; j < rulePred.labels.length; j++) {
      const lbl = rulePred.labels[j];
      if (seen.has(lbl)) continue;
      const rawScore = (rulePred.scores[j] || 0) * (rulePeak || 1);
      labels.push(lbl);
      scores.push(rawScore);
      ranked.push({ label: lbl, score: rawScore, isCustom: false });
    }

    ranked.sort(function (a, b) { return b.score - a.score; });

    const top = ranked[0];
    const topLabel = (top && top.score >= 0.25) ? top.label : 'unknown';
    const topScore = top ? top.score : 0;

    return {
      label: topLabel,
      score: topScore,
      labels: labels,
      scores: scores,
      ranked: ranked,
      measurements: rulePred.measurements
    };
  }

  function handleFeature(feature, landmarks) {
    if (!feature || !landmarks || landmarks.length < 33) {
      el.topLabel.textContent = '—';
      el.topScore.textContent = 'no pose detected';
      state.smoother.push(null);
      if (state.health) {
        renderPosture({ valid: false, label: 'No person detected', score: 0, advice: 'Step into camera frame' });
        renderRoutine(state.health.routine.getStatus());
        renderRom(state.health.rom.update(null));
      }
      return;
    }

    let prediction = null;
    const mode = el.backendMode ? el.backendMode.value : 'auto';

    if (mode === 'trained') {
      const backend = activeBackend();
      prediction = backend ? backend.predict(feature.values) : null;
      if (!prediction) {
        el.topLabel.textContent = '—';
        el.topScore.textContent = 'record a couple of poses first';
      }
    } else if (mode === 'rules') {
      prediction = PZ.rules.predict(feature);
      renderMeasurements(prediction.measurements);
    } else {
      // 'auto' mode: combine built-in rules and custom poses
      const rulePred = PZ.rules.predict(feature);
      renderMeasurements(rulePred.measurements);

      const customBackend = activeBackend();
      if (!customBackend || !customBackend.labels.length) {
        prediction = rulePred;
      } else {
        const customPred = customBackend.predict(feature.values);
        prediction = combinePredictions(rulePred, customPred);
      }
    }

    let activePose = null;
    let activeScore = 0;

    if (prediction) {
      const smoothed = state.smoother.push(feature.visible ? prediction : null);
      const shown = feature.visible ? (smoothed.label || prediction.label) : 'out of frame';
      el.topLabel.textContent = shown;
      el.topScore.textContent = feature.visible
        ? Math.round(smoothed.score * 100) + '% confident'
        : 'step back so your whole body is visible';
      renderScores(prediction.labels, smoothed.scores || prediction.scores, shown);
      renderCounts(smoothed.counts);

      if (feature.visible) {
        activePose = smoothed.label || prediction.label;
        activeScore = smoothed.score || prediction.score;
      }
    }

    if (state.recording) captureSample(feature, landmarks);

    // Health suite updates
    if (state.health && landmarks && landmarks.length >= 33) {
      state.health.lastLandmarks = landmarks;

      const postResult = state.health.posture.analyze(landmarks);
      renderPosture(postResult);
      if (postResult.alert && el.postureAlertToggle && el.postureAlertToggle.checked) {
        state.health.routine.playChime('rep');
      }

      const routineResult = state.health.routine.update(activePose, activeScore);
      renderRoutine(routineResult);

      const romResult = state.health.rom.update(landmarks);
      renderRom(romResult);
    }
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

  /* ---------- health rendering helpers ---------- */

  function renderPosture(res) {
    if (!el.postureLabel || !res) return;
    if (!res.valid) {
      el.postureLabel.textContent = res.label || 'No person detected';
      el.postureAdvice.textContent = res.advice || 'Step into camera frame';
      el.postureScoreNum.textContent = '—';
      el.postureReadout.className = 'posture-readout';
      return;
    }

    el.postureScoreNum.textContent = res.score;
    el.postureLabel.textContent = res.label;
    el.postureAdvice.textContent = res.advice;

    el.postureReadout.className = 'posture-readout status-' + (res.status || 'good');
    if (res.metrics) {
      el.postureHeadVal.textContent = res.metrics.headAngle + '°';
      el.postureSpineVal.textContent = res.metrics.spineAngle + '°';
      el.postureShoulderVal.textContent = res.metrics.shoulderTilt + '°';
    }
  }

  function renderRoutine(res) {
    if (!el.routineStepName || !res) return;
    if (res.state === 'idle') {
      el.routineStepBadge.textContent = 'Ready';
      el.routineRepBadge.textContent = 'Rep 0 / 0';
      el.routineStepName.textContent = 'Select a routine';
      el.routineCue.textContent = 'Click "Start routine" to begin guided therapy.';
      el.routineHoldTimer.textContent = '0.0s';
      el.routineProgressBar.style.width = '0%';
      el.routineToggleBtn.textContent = 'Start routine';
      return;
    }

    if (res.state === 'finished') {
      el.routineStepBadge.textContent = 'Complete';
      el.routineRepBadge.textContent = 'Done';
      el.routineStepName.textContent = 'Routine Finished!';
      el.routineCue.textContent = 'Great work! You completed all exercises with good form.';
      el.routineHoldTimer.textContent = '100%';
      el.routineProgressBar.style.width = '100%';
      el.routineToggleBtn.textContent = 'Restart routine';
      return;
    }

    const step = res.step;
    el.routineStepBadge.textContent = 'Step ' + (res.stepIndex + 1) + ' / ' + res.totalSteps;
    el.routineRepBadge.textContent = 'Rep ' + res.currentRep + ' / ' + res.targetReps;
    if (step) {
      el.routineStepName.textContent = step.label;
      el.routineCue.textContent = (res.isMatching ? '✓ Good form! ' : 'Hold pose: ') + step.cue;
      el.routineHoldTimer.textContent = res.elapsedHoldSec.toFixed(1) + 's / ' + res.holdSec + '.0s';
      el.routineProgressBar.style.width = res.progressPercent + '%';
    }
    el.routineToggleBtn.textContent = 'Stop routine';
  }

  function renderRom(res) {
    if (!el.romCurrentDeg || !res) return;
    el.romCurrentDeg.textContent = res.currentDeg + '°';
    el.romPeakDeg.textContent = res.sessionPeakDeg + '°';
    el.romNormLabel.textContent = 'Peak (norm: ' + res.normalDeg + '°)';
    el.romProgressFill.style.width = res.percentOfNormal + '%';

    if (res.testActive) {
      el.romTestBtn.textContent = 'Testing (' + res.testProgress + '%)';
      el.romTestBtn.disabled = true;
      el.romTestStatus.textContent = 'Measuring peak range: ' + res.testPeakDeg + '°...';
    } else {
      el.romTestBtn.textContent = 'Start 5s mobility test';
      el.romTestBtn.disabled = false;
      if (res.completedTestResult) {
        const r = res.completedTestResult;
        el.romTestStatus.textContent = 'Result: ' + r.peakDeg + '° (' + r.percentNormal + '% normal) — ' + r.grade;
      }
    }
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
    state.renderedLabels = null;
    if (el.backendMode.value === 'rules') {
      el.backendMode.value = 'auto';
    }
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
    const data = state.dataset.toTraining(3);
    if (data.labels.length < 1) { state.knn = null; return; }
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
        state.renderedLabels = null;
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
    try {
      global.localStorage.removeItem(MODEL_KEY);
    } catch (err) { /* nothing stored */ }
    renderClasses();
    setTrainStatus('All samples cleared.');
  });
  el.exportData.addEventListener('click', function () { download('possess-samples.json', state.dataset.toJSON()); });
  el.exportModel.addEventListener('click', function () {
    if (!state.model) { setTrainStatus('Train a model before exporting it.', true); return; }
    download('possess-model.json', state.model.toJSON());
  });
  el.importData.addEventListener('change', async function (e) {
    const file = e.target.files[0];
    if (!file) return;
    try {
      state.dataset = PZ.dataset.PoseDataset.fromJSON(await readJson(file));
      persistDataset();
      rebuildKnn();
      renderClasses();
      state.renderedLabels = null;
      if (el.backendMode.value === 'rules') {
        el.backendMode.value = 'auto';
      }
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

  /* ---------- modular dashboard widgets ---------- */

  const WIDGET_STORAGE_KEY = 'possess.widgets.v3';
  const DEFAULT_LAYOUT = {
    order: ['source', 'prediction', 'teach', 'posture', 'routine', 'rom'],
    cols: { source: 5, prediction: 3, teach: 4, posture: 4, routine: 4, rom: 4 }
  };

  function initWidgets() {
    const grid = doc.getElementById('dashboardGrid');
    if (!grid) return;

    function loadLayout() {
      try {
        const raw = global.localStorage.getItem(WIDGET_STORAGE_KEY);
        if (raw) {
          const parsed = JSON.parse(raw);
          if (parsed && Array.isArray(parsed.order) && parsed.cols) return parsed;
        }
      } catch (err) { /* ignore */ }
      return DEFAULT_LAYOUT;
    }

    function saveLayout() {
      try {
        const widgets = Array.from(grid.querySelectorAll('.widget'));
        const order = widgets.map(function (w) { return w.dataset.widget; });
        const cols = {};
        widgets.forEach(function (w) { cols[w.dataset.widget] = parseInt(w.dataset.cols, 10) || 12; });
        global.localStorage.setItem(WIDGET_STORAGE_KEY, JSON.stringify({ order: order, cols: cols }));
      } catch (err) { /* ignore */ }
    }

    function applyLayout(layout) {
      const widgetsById = {};
      grid.querySelectorAll('.widget').forEach(function (w) {
        widgetsById[w.dataset.widget] = w;
      });

      layout.order.forEach(function (id) {
        const w = widgetsById[id];
        if (w) {
          grid.appendChild(w);
          const col = layout.cols[id] || (DEFAULT_LAYOUT.cols && DEFAULT_LAYOUT.cols[id]) || 12;
          w.dataset.cols = col;
          delete widgetsById[id];
        }
      });

      // Append any remaining widgets that were not in saved layout
      Object.keys(widgetsById).forEach(function (id) {
        const w = widgetsById[id];
        grid.appendChild(w);
        const col = (DEFAULT_LAYOUT.cols && DEFAULT_LAYOUT.cols[id]) || 4;
        w.dataset.cols = col;
      });
    }

    // Initialize layout
    applyLayout(loadLayout());

    // Reset layout button
    const resetBtn = doc.getElementById('resetLayout');
    if (resetBtn) {
      resetBtn.addEventListener('click', function () {
        global.localStorage.removeItem(WIDGET_STORAGE_KEY);
        applyLayout(DEFAULT_LAYOUT);
      });
    }

    // Drag & Drop reordering
    let draggedWidget = null;

    function isDropBefore(targetWidget, clientX, clientY) {
      const rect = targetWidget.getBoundingClientRect();
      const gridRect = grid.getBoundingClientRect();
      // If single column / mobile or nearly full width, determine by vertical position
      if (rect.width > gridRect.width * 0.75) {
        return clientY < rect.top + rect.height / 2;
      }
      // In a multi-column row, determine by horizontal position
      return clientX < rect.left + rect.width / 2;
    }

    grid.querySelectorAll('.widget').forEach(function (widget) {
      const handle = widget.querySelector('.drag-handle');
      if (handle) {
        handle.addEventListener('mouseenter', function () { widget.draggable = true; });
        handle.addEventListener('mouseleave', function () { if (!draggedWidget) widget.draggable = false; });
      }

      widget.addEventListener('dragstart', function (e) {
        draggedWidget = widget;
        widget.classList.add('is-dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', widget.dataset.widget);
      });

      widget.addEventListener('dragend', function () {
        widget.classList.remove('is-dragging');
        draggedWidget = null;
        widget.draggable = false;
        grid.querySelectorAll('.widget').forEach(function (w) {
          w.classList.remove('drag-over-before', 'drag-over-after');
        });
        saveLayout();
      });

      widget.addEventListener('dragover', function (e) {
        if (!draggedWidget || draggedWidget === widget) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';

        if (isDropBefore(widget, e.clientX, e.clientY)) {
          widget.classList.add('drag-over-before');
          widget.classList.remove('drag-over-after');
        } else {
          widget.classList.add('drag-over-after');
          widget.classList.remove('drag-over-before');
        }
      });

      widget.addEventListener('dragleave', function () {
        widget.classList.remove('drag-over-before', 'drag-over-after');
      });

      widget.addEventListener('drop', function (e) {
        if (!draggedWidget || draggedWidget === widget) return;
        e.preventDefault();
        if (isDropBefore(widget, e.clientX, e.clientY)) {
          grid.insertBefore(draggedWidget, widget);
        } else {
          grid.insertBefore(draggedWidget, widget.nextSibling);
        }
        widget.classList.remove('drag-over-before', 'drag-over-after');
        saveLayout();
      });

      // Drag-to-resize corner handle
      const resizeHandle = widget.querySelector('.widget-resize-handle');
      if (resizeHandle) {
        resizeHandle.addEventListener('pointerdown', function (e) {
          e.preventDefault();
          resizeHandle.setPointerCapture(e.pointerId);
          widget.classList.add('is-resizing');

          const startX = e.clientX;
          const startCols = parseInt(widget.dataset.cols, 10) || 12;
          const gridWidth = grid.getBoundingClientRect().width;
          const oneColWidth = gridWidth / 12;

          function onPointerMove(moveEvent) {
            const deltaX = moveEvent.clientX - startX;
            const colDelta = Math.round(deltaX / oneColWidth);
            const targetCols = Math.min(12, Math.max(2, startCols + colDelta));
            if (parseInt(widget.dataset.cols, 10) !== targetCols) {
              widget.dataset.cols = targetCols;
            }
          }

          function onPointerUp(upEvent) {
            resizeHandle.releasePointerCapture(upEvent.pointerId);
            resizeHandle.removeEventListener('pointermove', onPointerMove);
            resizeHandle.removeEventListener('pointerup', onPointerUp);
            widget.classList.remove('is-resizing');
            saveLayout();
          }

          resizeHandle.addEventListener('pointermove', onPointerMove);
          resizeHandle.addEventListener('pointerup', onPointerUp);
        });
      }
    });
  }

  /* ---------- health event listeners ---------- */

  if (el.calibratePosture) {
    el.calibratePosture.addEventListener('click', function () {
      if (state.health && state.health.lastLandmarks) {
        const base = state.health.posture.calibrateBaseline(state.health.lastLandmarks);
        if (base) {
          el.calibratePosture.textContent = 'Calibrated ✓';
          setTimeout(function () { el.calibratePosture.textContent = 'Calibrate'; }, 2000);
        }
      } else {
        el.calibratePosture.textContent = 'Stand in frame first';
        setTimeout(function () { el.calibratePosture.textContent = 'Calibrate'; }, 2000);
      }
    });
  }

  if (el.routineToggleBtn) {
    el.routineToggleBtn.addEventListener('click', function () {
      if (!state.health) return;
      if (state.health.routine.state === 'idle' || state.health.routine.state === 'finished') {
        state.health.routine.start(el.routineSelect ? el.routineSelect.value : 'desk-reset');
      } else {
        state.health.routine.stop();
      }
      renderRoutine(state.health.routine.getStatus());
    });
  }

  if (el.routineResetBtn) {
    el.routineResetBtn.addEventListener('click', function () {
      if (!state.health) return;
      state.health.routine.stop();
      renderRoutine(state.health.routine.getStatus());
    });
  }

  if (el.routineSelect) {
    el.routineSelect.addEventListener('change', function () {
      if (!state.health) return;
      if (state.health.routine.state !== 'idle') {
        state.health.routine.start(el.routineSelect.value);
      }
      renderRoutine(state.health.routine.getStatus());
    });
  }

  if (el.romJointSelect) {
    el.romJointSelect.addEventListener('change', function () {
      if (!state.health) return;
      state.health.rom.setJoint(el.romJointSelect.value);
      renderRom(state.health.rom.update(state.health.lastLandmarks));
    });
  }

  if (el.romTestBtn) {
    el.romTestBtn.addEventListener('click', function () {
      if (!state.health) return;
      state.health.rom.startTest(5);
      renderRom(state.health.rom.update(state.health.lastLandmarks));
    });
  }

  if (el.exportHealthReport) {
    el.exportHealthReport.addEventListener('click', function () {
      if (!state.health) return;
      const json = state.health.session.exportJSON(
        state.health.posture,
        state.health.routine,
        state.health.rom
      );
      download('possess-health-report.json', json);
    });
  }

  /* ---------- start ---------- */

  populateModels();
  refreshCacheStatus();
  loadSavedModel();
  rebuildKnn();
  renderClasses();
  renderCounts({});
  if (state.model) el.backendMode.value = 'trained';
  showStage('none');
  initWidgets();

  if (state.health) {
    renderPosture({ valid: false, label: 'Ready', score: 100, advice: 'Stand in frame to monitor posture' });
    renderRoutine(state.health.routine.getStatus());
    renderRom(state.health.rom.update(null));
  }

  // Exposed for the end-to-end test, which drives the page without a camera.
  global.__possess = state;
})(typeof globalThis !== 'undefined' ? globalThis : this);
