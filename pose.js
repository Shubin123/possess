/*
 * possess - the pose estimator.
 *
 * Wraps MediaPipe's Pose Landmarker (BlazePose GHUM), the successor to the
 * PoseNet/MoveNet line: 33 landmarks per person, running on WebGL or WASM in
 * the browser. The task bundle and the weights come from public CDNs on first
 * load and are then served from IndexedDB; no application server exists.
 *
 * The bundle is pulled in with a dynamic `import()` so the URL stays a plain
 * configuration value - which is what lets the end-to-end test swap in a
 * deterministic fake landmarker without touching the network.
 */
(function (global) {
  const PZ = (global.PZ = global.PZ || {});

  const VISION_VERSION = '1.0.1';
  const BUNDLE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@' + VISION_VERSION + '/vision_bundle.mjs';
  const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@' + VISION_VERSION + '/wasm';
  const MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/';

  // The three published variants, cheapest first. `bytes` is the real download
  // size, shown in the UI before the user commits to fetching one.
  const MODELS = {
    lite: {
      id: 'lite',
      label: 'Pose Landmarker Lite',
      url: MODEL_BASE + 'pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
      bytes: 5777746,
      note: 'Fastest. Enough for whole-body poses on a laptop webcam.'
    },
    full: {
      id: 'full',
      label: 'Pose Landmarker Full',
      url: MODEL_BASE + 'pose_landmarker_full/float16/1/pose_landmarker_full.task',
      bytes: 9398198,
      note: 'Steadier landmarks, still real time on most machines.'
    },
    heavy: {
      id: 'heavy',
      label: 'Pose Landmarker Heavy',
      url: MODEL_BASE + 'pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
      bytes: 30664242,
      note: 'Most accurate, noticeably slower without a GPU.'
    }
  };

  /*
   * Loads a detector. `onProgress` reports the weight download; `delegate`
   * picks GPU or CPU inference and falls back to CPU when WebGL is missing.
   */
  async function createDetector(options) {
    const opts = options || {};
    const spec = MODELS[opts.model || 'lite'] || MODELS.lite;
    const bundleUrl = opts.bundleUrl || BUNDLE_URL;
    const wasmBase = opts.wasmBase || WASM_BASE;

    const vision = await import(/* webpackIgnore: true */ bundleUrl);
    const fileset = await vision.FilesetResolver.forVisionTasks(wasmBase);
    const modelBytes = await PZ.modelCache.fetchModel(spec.url, opts.onProgress);

    const baseConfig = {
      runningMode: 'VIDEO',
      numPoses: opts.numPoses || 1,
      minPoseDetectionConfidence: opts.minDetectionConfidence || 0.5,
      minPosePresenceConfidence: opts.minPresenceConfidence || 0.5,
      minTrackingConfidence: opts.minTrackingConfidence || 0.5,
      outputSegmentationMasks: false
    };

    let landmarker;
    let delegate = opts.delegate || 'GPU';
    try {
      landmarker = await vision.PoseLandmarker.createFromOptions(fileset, Object.assign({
        baseOptions: { modelAssetBuffer: modelBytes, delegate: delegate }
      }, baseConfig));
    } catch (err) {
      if (delegate === 'CPU') throw err;
      delegate = 'CPU';
      landmarker = await vision.PoseLandmarker.createFromOptions(fileset, Object.assign({
        baseOptions: { modelAssetBuffer: modelBytes, delegate: 'CPU' }
      }, baseConfig));
    }

    let runningMode = 'VIDEO';
    const ensureMode = async function (mode) {
      if (runningMode === mode) return;
      await landmarker.setOptions({ runningMode: mode });
      runningMode = mode;
    };

    return {
      spec: spec,
      delegate: delegate,
      /*
       * Detects on a video frame (`timestampMs` strictly increasing) or, when
       * no timestamp is given, on a still image or canvas. Returns the first
       * pose's landmarks plus how long inference took.
       */
      detect: async function (source, timestampMs) {
        const started = (global.performance || Date).now();
        let result;
        if (timestampMs === undefined || timestampMs === null) {
          await ensureMode('IMAGE');
          result = landmarker.detect(source);
        } else {
          await ensureMode('VIDEO');
          result = landmarker.detectForVideo(source, timestampMs);
        }
        const landmarks = result && result.landmarks && result.landmarks.length ? result.landmarks[0] : null;
        return {
          landmarks: landmarks,
          worldLandmarks: result && result.worldLandmarks && result.worldLandmarks.length ? result.worldLandmarks[0] : null,
          latencyMs: (global.performance || Date).now() - started
        };
      },
      close: function () { try { landmarker.close(); } catch (err) { /* already gone */ } }
    };
  }

  PZ.pose = {
    MODELS: MODELS,
    BUNDLE_URL: BUNDLE_URL,
    WASM_BASE: WASM_BASE,
    createDetector: createDetector
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
