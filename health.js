/*
 * possess - Health, Ergonomics & Physical Therapy Engine.
 *
 * 1. Posture & Ergonomics Guard:
 *    - Real-time forward head angle (ear-to-shoulder offset)
 *    - Torso slouch & spinal alignment
 *    - Lateral shoulder tilt & pelvic level (asymmetry)
 *    - Posture Score (0-100%), classification (good, mild, poor), slouch timer
 *    - Upright baseline calibration
 *
 * 2. Guided Exercise & Therapy Routines:
 *    - Curated physical therapy & mobility routines (Desk Reset, Lower Body, Core)
 *    - Hold timer & cadence state machine
 *    - Real-time form verification
 *    - Web Audio synthesized audio coaching chimes
 *
 * 3. Range of Motion (ROM) Clinical Goniometer:
 *    - Target joints: Shoulders, Elbows, Knees, Hips, Trunk
 *    - AAOS clinical norm comparisons
 *    - Peak ROM tracking & 5-second test assessment grading
 *
 * 4. Health Session Analytics & Export:
 *    - Session timeline, posture score %, completed exercises, ROM logs
 *    - Privacy-first export for clinical or personal health tracking
 *
 * Pure and dependency-free: works in browser and Node.js test environment.
 */
(function (global) {
  const PZ = (global.PZ = global.PZ || {});
  const features = PZ.features || (typeof require !== 'undefined' ? require('./features.js') : null);

  const LM = features ? features.LM : {
    NOSE: 0,
    L_EAR: 7, R_EAR: 8,
    L_SHOULDER: 11, R_SHOULDER: 12,
    L_ELBOW: 13, R_ELBOW: 14,
    L_WRIST: 15, R_WRIST: 16,
    L_HIP: 23, R_HIP: 24,
    L_KNEE: 25, R_KNEE: 26,
    L_ANKLE: 27, R_ANKLE: 28,
    L_FOOT: 31, R_FOOT: 32
  };

  const toDeg = rad => rad * (180 / Math.PI);
  const toRad = deg => deg * (Math.PI / 180);

  function mid(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  }

  function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function interiorAngleDeg(a, b, c) {
    const v1x = a.x - b.x, v1y = a.y - b.y;
    const v2x = c.x - b.x, v2y = c.y - b.y;
    const n1 = Math.hypot(v1x, v1y), n2 = Math.hypot(v2x, v2y);
    if (n1 < 1e-9 || n2 < 1e-9) return 0;
    const cos = (v1x * v2x + v1y * v2y) / (n1 * n2);
    return toDeg(Math.acos(Math.min(1, Math.max(-1, cos))));
  }

  /* =========================================================================
   * 1. Posture & Ergonomics Engine
   * ========================================================================= */

  class PostureMonitor {
    constructor() {
      this.baseline = null;
      this.slouchDurationSec = 0;
      this.alertTriggered = false;
      this.history = { good: 0, mild: 0, poor: 0 };
    }

    calibrateBaseline(landmarks) {
      if (!landmarks || landmarks.length < 33) return null;
      const lEar = landmarks[LM.L_EAR] || landmarks[7];
      const rEar = landmarks[LM.R_EAR] || landmarks[8];
      const lSh = landmarks[LM.L_SHOULDER] || landmarks[11];
      const rSh = landmarks[LM.R_SHOULDER] || landmarks[12];
      const lHip = landmarks[LM.L_HIP] || landmarks[23];
      const rHip = landmarks[LM.R_HIP] || landmarks[24];

      const earMid = mid(lEar, rEar);
      const shMid = mid(lSh, rSh);
      const hipMid = mid(lHip, rHip);

      this.baseline = {
        torsoLength: dist(shMid, hipMid),
        shoulderWidth: dist(lSh, rSh),
        earToShoulderY: Math.abs(shMid.y - earMid.y),
        calibratedAt: Date.now()
      };
      return this.baseline;
    }

    clearBaseline() {
      this.baseline = null;
    }

    analyze(landmarks, dtSec = 0.05) {
      if (!landmarks || landmarks.length < 33) {
        return {
          valid: false,
          status: 'no_person',
          score: 0,
          label: 'No person detected'
        };
      }

      const lEar = landmarks[LM.L_EAR] || landmarks[7];
      const rEar = landmarks[LM.R_EAR] || landmarks[8];
      const lSh = landmarks[LM.L_SHOULDER] || landmarks[11];
      const rSh = landmarks[LM.R_SHOULDER] || landmarks[12];
      const lHip = landmarks[LM.L_HIP] || landmarks[23];
      const rHip = landmarks[LM.R_HIP] || landmarks[24];
      const nose = landmarks[LM.NOSE] || landmarks[0];

      // Check visibility of upper body landmarks
      const upperPts = [lSh, rSh, nose];
      const avgVis = upperPts.reduce((acc, p) => acc + (p.visibility ?? 1), 0) / upperPts.length;
      if (avgVis < 0.4) {
        return {
          valid: false,
          status: 'out_of_frame',
          score: 0,
          label: 'Upper body out of frame'
        };
      }

      const earMid = mid(lEar, rEar);
      const shMid = mid(lSh, rSh);
      const hipMid = mid(lHip, rHip);

      const shWidth = Math.max(1e-4, dist(lSh, rSh));
      const torsoLen = Math.max(1e-4, dist(shMid, hipMid));

      // 1. Forward head offset (angle from vertical)
      const headDx = earMid.x - shMid.x;
      const headDy = Math.max(1e-4, Math.abs(shMid.y - earMid.y));
      const headAngle = toDeg(Math.atan2(Math.abs(headDx), headDy));

      // 2. Torso slouch & spinal tilt
      const torsoDx = shMid.x - hipMid.x;
      const torsoDy = Math.max(1e-4, Math.abs(hipMid.y - shMid.y));
      const spineAngle = toDeg(Math.atan2(Math.abs(torsoDx), torsoDy));

      // Torso compression relative to baseline (if calibrated)
      let torsoCompression = 1.0;
      if (this.baseline && this.baseline.torsoLength > 1e-4) {
        torsoCompression = torsoLen / this.baseline.torsoLength;
      }

      // 3. Lateral shoulder tilt (levelness)
      const shDeltaY = rSh.y - lSh.y;
      const shTiltDeg = toDeg(Math.asin(Math.min(1, Math.max(-1, Math.abs(shDeltaY) / shWidth))));

      // 4. Scoring calculation (0 - 100)
      let score = 100;

      // Penalize forward head posture (> 10° is mild, > 20° is poor)
      if (headAngle > 10) {
        score -= (headAngle - 10) * 2.2;
      }

      // Penalize torso slouch / angle (> 8° is leaning/slouched)
      if (spineAngle > 8) {
        score -= (spineAngle - 8) * 2.5;
      }

      // Penalize torso compression if calibrated (< 90% upright)
      if (torsoCompression < 0.92) {
        score -= (0.92 - torsoCompression) * 120;
      }

      // Penalize shoulder asymmetry / tilt (> 4° is uneven)
      if (shTiltDeg > 4) {
        score -= (shTiltDeg - 4) * 3.0;
      }

      score = Math.max(0, Math.min(100, Math.round(score)));

      // Classification & state tracking
      let status = 'good';
      let label = 'Good posture';
      let advice = 'Spine, neck and shoulders aligned';

      if (score < 60) {
        status = 'poor';
        label = 'Slouched posture';
        if (headAngle > 18) advice = 'Pull head back, align ears over shoulders';
        else if (torsoCompression < 0.88 || spineAngle > 15) advice = 'Straighten spine, sit or stand tall';
        else if (shTiltDeg > 7) advice = 'Level your shoulders';
        else advice = 'Adjust seating and lift your chest';
      } else if (score < 80) {
        status = 'mild';
        label = 'Mild slouch';
        advice = 'Gently roll shoulders back and lift chin';
      }

      // Slouch timer tracking
      if (status === 'poor') {
        this.slouchDurationSec += dtSec;
        if (this.slouchDurationSec >= 10 && !this.alertTriggered) {
          this.alertTriggered = true;
        }
      } else {
        this.slouchDurationSec = Math.max(0, this.slouchDurationSec - dtSec * 2);
        this.alertTriggered = false;
      }

      // Health session history
      this.history[status] = (this.history[status] || 0) + 1;

      return {
        valid: true,
        status: status,
        score: score,
        label: label,
        advice: advice,
        metrics: {
          headAngle: Math.round(headAngle),
          spineAngle: Math.round(spineAngle),
          shoulderTilt: Math.round(shTiltDeg),
          torsoCompression: Math.round(torsoCompression * 100)
        },
        slouchDurationSec: Math.round(this.slouchDurationSec),
        alert: this.alertTriggered,
        hasBaseline: !!this.baseline
      };
    }
  }

  /* =========================================================================
   * 2. Guided Exercise & Therapy Routines
   * ========================================================================= */

  const PRESET_ROUTINES = [
    {
      id: 'desk-reset',
      title: 'Desk Ergonomics Reset',
      description: 'Quick 2-minute spinal extension & shoulder mobility.',
      steps: [
        { pose: 'standing', label: 'Upright Posture', holdSec: 4, reps: 3, cue: 'Stand or sit tall, shoulders relaxed' },
        { pose: 't-pose', label: 'T-Pose Chest Opener', holdSec: 5, reps: 3, cue: 'Extend arms horizontal, squeeze shoulder blades' },
        { pose: 'hands-up', label: 'Overhead Reach', holdSec: 5, reps: 3, cue: 'Reach arms straight up to decompress spine' }
      ]
    },
    {
      id: 'lower-mobility',
      title: 'Lower Body & Hip Mobility',
      description: 'Squat depth hold and unilateral hip stabilizer balance.',
      steps: [
        { pose: 'squat', label: 'Squat Hold', holdSec: 3, reps: 5, cue: 'Sink hips back, chest high, knees outward' },
        { pose: 'one-arm-up-left', label: 'Left Side Balance', holdSec: 4, reps: 3, cue: 'Engage core, lift left arm, hold steady' },
        { pose: 'one-arm-up-right', label: 'Right Side Balance', holdSec: 4, reps: 3, cue: 'Engage core, lift right arm, hold steady' }
      ]
    },
    {
      id: 'core-balance',
      title: 'Shoulder & Core Stability',
      description: 'Isometric holds for postural stamina and balance.',
      steps: [
        { pose: 't-pose', label: 'T-Pose Holds', holdSec: 6, reps: 4, cue: 'Maintain level arms without dipping' },
        { pose: 'hands-up', label: 'Sky Reach Hold', holdSec: 6, reps: 4, cue: 'Keep core tight and arms extended vertically' }
      ]
    }
  ];

  class RoutineEngine {
    constructor(audioFeedback = true) {
      this.routines = PRESET_ROUTINES;
      this.currentRoutine = null;
      this.stepIndex = 0;
      this.currentRep = 1;
      this.elapsedHoldSec = 0;
      this.state = 'idle'; // 'idle' | 'holding' | 'rep_completed' | 'finished'
      this.audioEnabled = audioFeedback;
      this.audioCtx = null;
      this.completedRoutinesCount = 0;
    }

    start(routineId) {
      const found = this.routines.find(r => r.id === routineId) || this.routines[0];
      this.currentRoutine = found;
      this.stepIndex = 0;
      this.currentRep = 1;
      this.elapsedHoldSec = 0;
      this.state = 'holding';
      return this.getStatus();
    }

    stop() {
      this.state = 'idle';
      this.currentRoutine = null;
      this.stepIndex = 0;
      this.currentRep = 1;
      this.elapsedHoldSec = 0;
    }

    getCurrentStep() {
      if (!this.currentRoutine) return null;
      return this.currentRoutine.steps[this.stepIndex] || null;
    }

    update(activePoseName, confidence = 0, dtSec = 0.05) {
      if (this.state === 'idle' || !this.currentRoutine) {
        return this.getStatus();
      }

      const step = this.getCurrentStep();
      if (!step) {
        this.state = 'finished';
        return this.getStatus();
      }

      // Check if user is holding required pose
      const isMatching = activePoseName === step.pose && confidence >= 0.45;

      if (isMatching) {
        this.elapsedHoldSec += dtSec;
        if (this.elapsedHoldSec >= step.holdSec) {
          // Rep completed
          this.playChime('rep');
          if (this.currentRep < step.reps) {
            this.currentRep++;
            this.elapsedHoldSec = 0;
            this.state = 'rep_completed';
          } else {
            // Step completed, advance or finish
            if (this.stepIndex + 1 < this.currentRoutine.steps.length) {
              this.stepIndex++;
              this.currentRep = 1;
              this.elapsedHoldSec = 0;
              this.state = 'holding';
            } else {
              this.state = 'finished';
              this.completedRoutinesCount++;
              this.playChime('finish');
            }
          }
        } else {
          this.state = 'holding';
        }
      } else {
        // Decay hold progress if out of pose
        this.elapsedHoldSec = Math.max(0, this.elapsedHoldSec - dtSec * 1.5);
      }

      return this.getStatus(isMatching);
    }

    getStatus(isMatching = false) {
      const step = this.getCurrentStep();
      const progressPercent = step
        ? Math.min(100, Math.round((this.elapsedHoldSec / step.holdSec) * 100))
        : 0;

      return {
        state: this.state,
        routineTitle: this.currentRoutine ? this.currentRoutine.title : null,
        routineId: this.currentRoutine ? this.currentRoutine.id : null,
        stepIndex: this.stepIndex,
        totalSteps: this.currentRoutine ? this.currentRoutine.steps.length : 0,
        step: step,
        currentRep: this.currentRep,
        targetReps: step ? step.reps : 0,
        holdSec: step ? step.holdSec : 0,
        elapsedHoldSec: Number(this.elapsedHoldSec.toFixed(1)),
        progressPercent: progressPercent,
        isMatching: isMatching
      };
    }

    playChime(type = 'rep') {
      if (!this.audioEnabled || typeof window === 'undefined') return;
      try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;
        if (!this.audioCtx) this.audioCtx = new AudioContext();
        if (this.audioCtx.state === 'suspended') this.audioCtx.resume();

        const ctx = this.audioCtx;
        const now = ctx.currentTime;

        if (type === 'rep') {
          // Cheerful two-tone chime (C5 -> E5)
          [523.25, 659.25].forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, now + i * 0.12);
            gain.gain.setValueAtTime(0.001, now + i * 0.12);
            gain.gain.exponentialRampToValueAtTime(0.18, now + i * 0.12 + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.12 + 0.3);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now + i * 0.12);
            osc.stop(now + i * 0.12 + 0.32);
          });
        } else if (type === 'finish') {
          // Victorious harmonic triad (C5 -> E5 -> G5 -> C6)
          [523.25, 659.25, 783.99, 1046.5].forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(freq, now + i * 0.14);
            gain.gain.setValueAtTime(0.001, now + i * 0.14);
            gain.gain.exponentialRampToValueAtTime(0.2, now + i * 0.14 + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.14 + 0.55);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now + i * 0.14);
            osc.stop(now + i * 0.14 + 0.6);
          });
        }
      } catch (err) { /* audio unavailable */ }
    }
  }

  /* =========================================================================
   * 3. Range of Motion (ROM) Clinical Diagnostic Goniometer
   * ========================================================================= */

  const ROM_JOINTS = {
    'shoulder_l': { id: 'shoulder_l', name: 'Left Shoulder Elevation', normalDeg: 180, unit: 'deg' },
    'shoulder_r': { id: 'shoulder_r', name: 'Right Shoulder Elevation', normalDeg: 180, unit: 'deg' },
    'elbow_l': { id: 'elbow_l', name: 'Left Elbow Flexion', normalDeg: 145, unit: 'deg' },
    'elbow_r': { id: 'elbow_r', name: 'Right Elbow Flexion', normalDeg: 145, unit: 'deg' },
    'knee_l': { id: 'knee_l', name: 'Left Knee Flexion', normalDeg: 135, unit: 'deg' },
    'knee_r': { id: 'knee_r', name: 'Right Knee Flexion', normalDeg: 135, unit: 'deg' },
    'trunk_lateral': { id: 'trunk_lateral', name: 'Trunk Lateral Flexion', normalDeg: 35, unit: 'deg' }
  };

  class RomGoniometer {
    constructor() {
      this.selectedJoint = 'shoulder_r';
      this.sessionPeaks = {};
      this.testActive = false;
      this.testDurationSec = 5;
      this.testElapsedSec = 0;
      this.testPeakDeg = 0;
      this.recentAssessments = [];
    }

    setJoint(jointId) {
      if (ROM_JOINTS[jointId]) {
        this.selectedJoint = jointId;
      }
    }

    calculateJointAngle(jointId, landmarks) {
      if (!landmarks || landmarks.length < 33) return 0;
      const lSh = landmarks[LM.L_SHOULDER] || landmarks[11];
      const rSh = landmarks[LM.R_SHOULDER] || landmarks[12];
      const lEl = landmarks[LM.L_ELBOW] || landmarks[13];
      const rEl = landmarks[LM.R_ELBOW] || landmarks[14];
      const lWr = landmarks[LM.L_WRIST] || landmarks[15];
      const rWr = landmarks[LM.R_WRIST] || landmarks[16];
      const lHip = landmarks[LM.L_HIP] || landmarks[23];
      const rHip = landmarks[LM.R_HIP] || landmarks[24];
      const lKn = landmarks[LM.L_KNEE] || landmarks[25];
      const rKn = landmarks[LM.R_KNEE] || landmarks[26];
      const lAn = landmarks[LM.L_ANKLE] || landmarks[27];
      const rAn = landmarks[LM.R_ANKLE] || landmarks[28];

      switch (jointId) {
        case 'shoulder_l': {
          // Angle of arm (shoulder to wrist) relative to vertical hanging torso
          return Math.round(interiorAngleDeg(lHip, lSh, lWr));
        }
        case 'shoulder_r': {
          return Math.round(interiorAngleDeg(rHip, rSh, rWr));
        }
        case 'elbow_l': {
          // Clinical elbow flexion = 180° - interior angle
          const angle = interiorAngleDeg(lSh, lEl, lWr);
          return Math.round(Math.max(0, 180 - angle));
        }
        case 'elbow_r': {
          const angle = interiorAngleDeg(rSh, rEl, rWr);
          return Math.round(Math.max(0, 180 - angle));
        }
        case 'knee_l': {
          // Clinical knee flexion = 180° - interior angle
          const angle = interiorAngleDeg(lHip, lKn, lAn);
          return Math.round(Math.max(0, 180 - angle));
        }
        case 'knee_r': {
          const angle = interiorAngleDeg(rHip, rKn, rAn);
          return Math.round(Math.max(0, 180 - angle));
        }
        case 'trunk_lateral': {
          const shMid = mid(lSh, rSh);
          const hipMid = mid(lHip, rHip);
          const dx = shMid.x - hipMid.x;
          const dy = Math.max(1e-4, Math.abs(hipMid.y - shMid.y));
          return Math.round(toDeg(Math.atan2(Math.abs(dx), dy)));
        }
        default:
          return 0;
      }
    }

    startTest(durationSec = 5) {
      this.testActive = true;
      this.testDurationSec = durationSec;
      this.testElapsedSec = 0;
      this.testPeakDeg = 0;
    }

    stopTest() {
      this.testActive = false;
    }

    update(landmarks, dtSec = 0.05) {
      const spec = ROM_JOINTS[this.selectedJoint] || ROM_JOINTS['shoulder_r'];
      const currentDeg = this.calculateJointAngle(this.selectedJoint, landmarks);

      // Track session maximum peak
      const prevPeak = this.sessionPeaks[this.selectedJoint] || 0;
      if (currentDeg > prevPeak) {
        this.sessionPeaks[this.selectedJoint] = currentDeg;
      }

      let testResult = null;

      // Handle active assessment test
      if (this.testActive) {
        this.testElapsedSec += dtSec;
        if (currentDeg > this.testPeakDeg) {
          this.testPeakDeg = currentDeg;
        }

        if (this.testElapsedSec >= this.testDurationSec) {
          this.testActive = false;
          const percentNormal = Math.min(100, Math.round((this.testPeakDeg / spec.normalDeg) * 100));

          let grade = 'Normal mobility';
          let status = 'good';
          if (percentNormal < 50) {
            grade = 'Significant limitation';
            status = 'poor';
          } else if (percentNormal < 75) {
            grade = 'Moderate limitation';
            status = 'mild';
          } else if (percentNormal < 90) {
            grade = 'Mild limitation';
            status = 'mild';
          }

          testResult = {
            jointId: this.selectedJoint,
            jointName: spec.name,
            peakDeg: this.testPeakDeg,
            normalDeg: spec.normalDeg,
            percentNormal: percentNormal,
            grade: grade,
            status: status,
            timestamp: Date.now()
          };
          this.recentAssessments.unshift(testResult);
          if (this.recentAssessments.length > 5) this.recentAssessments.pop();
        }
      }

      const percentOfNormal = Math.min(100, Math.round((currentDeg / spec.normalDeg) * 100));

      return {
        jointId: this.selectedJoint,
        spec: spec,
        currentDeg: currentDeg,
        sessionPeakDeg: this.sessionPeaks[this.selectedJoint] || currentDeg,
        normalDeg: spec.normalDeg,
        percentOfNormal: percentOfNormal,
        testActive: this.testActive,
        testProgress: this.testActive ? Math.min(100, Math.round((this.testElapsedSec / this.testDurationSec) * 100)) : 0,
        testPeakDeg: this.testPeakDeg,
        completedTestResult: testResult,
        recentAssessments: this.recentAssessments
      };
    }
  }

  /* =========================================================================
   * 4. Health Session History & Export
   * ========================================================================= */

  class HealthSessionManager {
    constructor() {
      this.startTime = Date.now();
    }

    generateReport(postureMonitor, routineEngine, romGoniometer) {
      const durationSec = Math.round((Date.now() - this.startTime) / 1000);
      const post = postureMonitor.history;
      const totalPost = (post.good || 0) + (post.mild || 0) + (post.poor || 0);

      const goodPct = totalPost > 0 ? Math.round((post.good / totalPost) * 100) : 100;
      const mildPct = totalPost > 0 ? Math.round((post.mild / totalPost) * 100) : 0;
      const poorPct = totalPost > 0 ? Math.round((post.poor / totalPost) * 100) : 0;

      return {
        appName: 'possess',
        version: '0.1.0',
        generatedAt: new Date().toISOString(),
        sessionDurationMinutes: Number((durationSec / 60).toFixed(1)),
        postureAnalysis: {
          goodAlignmentPercent: goodPct,
          mildSlouchPercent: mildPct,
          poorPosturePercent: poorPct,
          hasUprightBaseline: !!postureMonitor.baseline
        },
        therapyRoutines: {
          routinesCompleted: routineEngine.completedRoutinesCount
        },
        rangeOfMotion: {
          sessionPeaks: romGoniometer.sessionPeaks,
          recentAssessments: romGoniometer.recentAssessments
        }
      };
    }

    exportJSON(postureMonitor, routineEngine, romGoniometer) {
      const report = this.generateReport(postureMonitor, routineEngine, romGoniometer);
      return JSON.stringify(report, null, 2);
    }
  }

  // Export health suite
  PZ.health = {
    PostureMonitor: PostureMonitor,
    RoutineEngine: RoutineEngine,
    RomGoniometer: RomGoniometer,
    HealthSessionManager: HealthSessionManager,
    PRESET_ROUTINES: PRESET_ROUTINES,
    ROM_JOINTS: ROM_JOINTS
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = PZ.health;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
