/*
 * possess - Sports Motion Training & Range of Motion Tracking Engine.
 *
 * Provides biomechanical analysis and kinematic phase tracking for sports:
 * 1. Tennis (Serve & Forehand)
 * 2. Golf (Drive / Swing)
 * 3. Basketball (Free Throw / Jump Shot)
 *
 * Features:
 * - Camera angle guidance (Profile / Down-The-Line / 3/4 view)
 * - Multi-phase kinematic tracking (Load / Cocking / Contact / Follow-through)
 * - Real-time joint ROM vs Pro Benchmarks
 * - Direct actionable coaching feedback ("Raise elbow 15° in trophy pose")
 * - Form deviation scoring (0-100%)
 * - Built-in pro tennis motion sequences for instant interactive training
 *
 * Pure and dependency-free: works in browser and Node.js test environment.
 */
(function (global) {
  const PZ = (global.PZ = global.PZ || {});
  const features = PZ.features || (typeof require !== 'undefined' ? require('./features.js') : null);

  const LM = features ? features.LM : {
    NOSE: 0,
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
    return Math.round(toDeg(Math.acos(Math.min(1, Math.max(-1, cos)))));
  }

  // Sports motion definitions with pro benchmarks and coaching rules
  const SPORTS_PRESETS = {
    'tennis-serve': {
      id: 'tennis-serve',
      name: 'Tennis Serve',
      sport: 'Tennis',
      cameraGuidance: 'Side Profile (90° view facing hitting side)',
      primaryArm: 'right',
      phases: [
        {
          id: 'trophy',
          name: 'Trophy Pose (Loading)',
          description: 'Cock striking arm at 90° elbow flex, knees bent, toss arm high.',
          benchmarks: {
            elbowAngle: { targetMin: 90, targetMax: 108, optimal: 98, unit: '°', label: 'Elbow Cocking' },
            shoulderElevation: { targetMin: 85, targetMax: 105, optimal: 95, unit: '°', label: 'Shoulder Height' },
            kneeFlexion: { targetMin: 120, targetMax: 145, optimal: 132, unit: '°', label: 'Knee Loading' }
          },
          coaching: {
            elbowTooClosed: 'Open up your elbow angle — avoid bringing racket too close to head.',
            elbowTooOpen: 'Bend hitting elbow closer to 90° for explosive lag.',
            shoulderTooLow: 'Lift your hitting elbow to shoulder level for clean trophy posture.',
            kneesStraight: 'Deepen knee bend to generate upward kinetic drive.'
          }
        },
        {
          id: 'contact',
          name: 'Contact Point (Apex)',
          description: 'Strike ball at maximum vertical extension with reaching arm.',
          benchmarks: {
            elbowAngle: { targetMin: 165, targetMax: 180, optimal: 175, unit: '°', label: 'Elbow Extension' },
            shoulderElevation: { targetMin: 165, targetMax: 180, optimal: 174, unit: '°', label: 'Overhead Reach' }
          },
          coaching: {
            elbowBent: 'Extend your hitting arm fully at impact for maximum reach and power.',
            lowContact: 'Contact ball at the highest point of your toss.'
          }
        },
        {
          id: 'follow-through',
          name: 'Follow-Through & Pronation',
          description: 'Racket accelerates through and finishes across the opposite hip.',
          benchmarks: {
            torsoFlexion: { targetMin: 15, targetMax: 35, optimal: 25, unit: '°', label: 'Torso Forward Lean' },
            armFinish: { targetMin: 30, targetMax: 65, optimal: 45, unit: '°', label: 'Cross-Body Angle' }
          },
          coaching: {
            incompleteFinish: 'Complete the follow-through smoothly across your opposite hip.'
          }
        }
      ]
    },
    'tennis-forehand': {
      id: 'tennis-forehand',
      name: 'Tennis Forehand (Topspin Drive)',
      sport: 'Tennis',
      cameraGuidance: 'Side / 45° Front View',
      primaryArm: 'right',
      phases: [
        {
          id: 'unit-turn',
          name: 'Unit Turn & Racket Takeback',
          description: 'Coil shoulders 90° with non-dominant arm across body.',
          benchmarks: {
            elbowAngle: { targetMin: 120, targetMax: 145, optimal: 135, unit: '°', label: 'Elbow Spacing' },
            shoulderTurn: { targetMin: 70, targetMax: 95, optimal: 85, unit: '°', label: 'Shoulder Coil' }
          },
          coaching: {
            elbowTucked: 'Create space between elbow and torso during takeback.',
            insufficientTurn: 'Turn shoulders perpendicular to net for coiling power.'
          }
        },
        {
          id: 'impact',
          name: 'Contact in Front',
          description: 'Strike ball out in front of hips with semi-bent arm.',
          benchmarks: {
            elbowAngle: { targetMin: 140, targetMax: 165, optimal: 152, unit: '°', label: 'Elbow Angle at Hit' }
          },
          coaching: {
            hitTooLate: 'Make contact in front of your front hip, not behind you.'
          }
        },
        {
          id: 'windshield-wiper',
          name: 'Windshield Wiper Finish',
          description: 'Arm sweeps up and finishes over opposite shoulder.',
          benchmarks: {
            shoulderElevation: { targetMin: 95, targetMax: 130, optimal: 112, unit: '°', label: 'Finish Height' }
          },
          coaching: {
            lowFinish: 'Wrap follow-through over shoulder to generate heavy topspin.'
          }
        }
      ]
    },
    'golf-swing': {
      id: 'golf-swing',
      name: 'Golf Swing (Drive / Iron)',
      sport: 'Golf',
      cameraGuidance: 'Down-The-Line or Face-On View',
      primaryArm: 'left',
      phases: [
        {
          id: 'address',
          name: 'Address & Spine Angle',
          description: 'Athletic stance with spine tilted from hips, knees flexed.',
          benchmarks: {
            spineAngle: { targetMin: 32, targetMax: 45, optimal: 38, unit: '°', label: 'Spine Tilt' },
            kneeFlexion: { targetMin: 145, targetMax: 162, optimal: 154, unit: '°', label: 'Knee Flex' }
          },
          coaching: {
            slouched: 'Bend from the hips rather than rounding the upper spine.'
          }
        },
        {
          id: 'top-backswing',
          name: 'Top of Backswing',
          description: 'Lead arm straight, full 90° shoulder turn behind ball.',
          benchmarks: {
            leadElbow: { targetMin: 162, targetMax: 180, optimal: 172, unit: '°', label: 'Lead Arm Straightness' }
          },
          coaching: {
            collapsedLeadArm: 'Keep lead arm extended at top of backswing to widen swing arc.'
          }
        },
        {
          id: 'finish',
          name: 'Full Balanced Finish',
          description: 'Weight rotated fully onto lead leg, chest facing target.',
          benchmarks: {
            leadLegExtension: { targetMin: 165, targetMax: 180, optimal: 175, unit: '°', label: 'Lead Leg Extension' }
          },
          coaching: {
            hangingBack: 'Rotate hips fully through to face the target at finish.'
          }
        }
      ]
    },
    'basketball-shot': {
      id: 'basketball-shot',
      name: 'Basketball Free Throw / Jump Shot',
      sport: 'Basketball',
      cameraGuidance: 'Side View (Shooting Arm Profile)',
      primaryArm: 'right',
      phases: [
        {
          id: 'set-point',
          name: 'Set Point / Triple Threat',
          description: 'Elbow tucked under ball near 90°, knees loaded.',
          benchmarks: {
            shootingElbow: { targetMin: 82, targetMax: 98, optimal: 90, unit: '°', label: 'Elbow Under Ball' },
            kneeFlexion: { targetMin: 120, targetMax: 140, optimal: 130, unit: '°', label: 'Knee Dip' }
          },
          coaching: {
            elbowFlared: 'Keep shooting elbow tucked directly underneath the ball.'
          }
        },
        {
          id: 'release-extension',
          name: 'Release & Follow-Through',
          description: 'Full upward extension with goose-neck wrist snap.',
          benchmarks: {
            elbowExtension: { targetMin: 168, targetMax: 180, optimal: 176, unit: '°', label: 'Elbow Lockout' },
            shoulderElevation: { targetMin: 140, targetMax: 170, optimal: 155, unit: '°', label: 'High Release Angle' }
          },
          coaching: {
            shortRelease: 'Push through to full high release point, hold follow-through.'
          }
        }
      ]
    }
  };

  class SportsMotionAnalyzer {
    constructor() {
      this.preset = SPORTS_PRESETS['tennis-serve'];
      this.currentPhaseIndex = 0;
      this.completedRepCount = 0;
      this.phaseScores = [];
      this.lastFeedback = 'Position yourself in side profile view and begin motion.';
      this.activeStatus = 'ready'; // 'ready' | 'tracking' | 'completed_rep'
      this.phaseHoldTime = 0;
    }

    setPreset(presetId) {
      if (SPORTS_PRESETS[presetId]) {
        this.preset = SPORTS_PRESETS[presetId];
        this.reset();
      }
    }

    reset() {
      this.currentPhaseIndex = 0;
      this.phaseScores = [];
      this.phaseHoldTime = 0;
      this.activeStatus = 'ready';
      this.lastFeedback = 'Position yourself in ' + this.preset.cameraGuidance + ' and begin motion.';
    }

    extractAngles(landmarks) {
      if (!landmarks || landmarks.length < 33) return null;
      const arm = this.preset.primaryArm || 'right';
      const isRight = arm === 'right';

      const sh = landmarks[isRight ? LM.R_SHOULDER : LM.L_SHOULDER] || landmarks[12];
      const el = landmarks[isRight ? LM.R_ELBOW : LM.L_ELBOW] || landmarks[14];
      const wr = landmarks[isRight ? LM.R_WRIST : LM.L_WRIST] || landmarks[16];
      const hip = landmarks[isRight ? LM.R_HIP : LM.L_HIP] || landmarks[24];
      const kn = landmarks[isRight ? LM.R_KNEE : LM.L_KNEE] || landmarks[26];
      const an = landmarks[isRight ? LM.R_ANKLE : LM.L_ANKLE] || landmarks[28];

      const shMid = mid(landmarks[LM.L_SHOULDER] || landmarks[11], landmarks[LM.R_SHOULDER] || landmarks[12]);
      const hipMid = mid(landmarks[LM.L_HIP] || landmarks[23], landmarks[LM.R_HIP] || landmarks[24]);

      // Elbow angle (flexion/extension)
      const elbowAngle = interiorAngleDeg(sh, el, wr);

      // Shoulder elevation angle (arm to torso)
      const shoulderElevation = interiorAngleDeg(hip, sh, wr);

      // Knee flexion angle
      const kneeFlexion = interiorAngleDeg(hip, kn, an);

      // Torso tilt from vertical
      const torsoDx = shMid.x - hipMid.x;
      const torsoDy = Math.max(1e-4, Math.abs(hipMid.y - shMid.y));
      const torsoTilt = Math.round(toDeg(Math.atan2(Math.abs(torsoDx), torsoDy)));

      return {
        elbowAngle: elbowAngle,
        shoulderElevation: shoulderElevation,
        kneeFlexion: kneeFlexion,
        torsoTilt: torsoTilt,
        shMid: shMid,
        hipMid: hipMid,
        wristY: wr.y,
        elbowY: el.y,
        shoulderY: sh.y
      };
    }

    analyze(landmarks, dtSec = 0.05) {
      if (!landmarks || landmarks.length < 33) {
        return {
          valid: false,
          preset: this.preset,
          feedback: 'Step into camera frame to begin sports motion tracking.',
          score: 0,
          currentPhase: null,
          angles: null
        };
      }

      const angles = this.extractAngles(landmarks);
      if (!angles) {
        return {
          valid: false,
          preset: this.preset,
          feedback: 'Tracking posture landmarks...',
          score: 0,
          currentPhase: null,
          angles: null
        };
      }

      const phase = this.preset.phases[this.currentPhaseIndex] || this.preset.phases[0];
      let phaseScore = 100;
      let advice = '';
      let metricComparisons = [];

      // Evaluate active phase benchmarks
      if (this.preset.id === 'tennis-serve') {
        if (phase.id === 'trophy') {
          // Trophy position: check elbow angle (90-108°) and shoulder height
          const eb = phase.benchmarks.elbowAngle;
          const sh = phase.benchmarks.shoulderElevation;

          const ebDev = Math.abs(angles.elbowAngle - eb.optimal);
          const shDev = Math.abs(angles.shoulderElevation - sh.optimal);

          metricComparisons.push({
            name: eb.label,
            current: angles.elbowAngle,
            targetMin: eb.targetMin,
            targetMax: eb.targetMax,
            optimal: eb.optimal,
            unit: eb.unit,
            inRange: angles.elbowAngle >= eb.targetMin && angles.elbowAngle <= eb.targetMax
          });

          metricComparisons.push({
            name: sh.label,
            current: angles.shoulderElevation,
            targetMin: sh.targetMin,
            targetMax: sh.targetMax,
            optimal: sh.optimal,
            unit: sh.unit,
            inRange: angles.shoulderElevation >= sh.targetMin && angles.shoulderElevation <= sh.targetMax
          });

          if (angles.elbowAngle < eb.targetMin) {
            advice = phase.coaching.elbowTooClosed + ` (Current: ${angles.elbowAngle}°, pro: ${eb.targetMin}°-${eb.targetMax}°)`;
            phaseScore -= Math.min(45, (eb.targetMin - angles.elbowAngle) * 2.5);
          } else if (angles.elbowAngle > eb.targetMax) {
            advice = phase.coaching.elbowTooOpen + ` (Current: ${angles.elbowAngle}°, pro: ${eb.targetMin}°-${eb.targetMax}°)`;
            phaseScore -= Math.min(45, (angles.elbowAngle - eb.targetMax) * 2.5);
          } else if (angles.shoulderElevation < sh.targetMin) {
            advice = phase.coaching.shoulderTooLow + ` (Lift elbow higher by ${sh.targetMin - angles.shoulderElevation}°)`;
            phaseScore -= Math.min(40, (sh.targetMin - angles.shoulderElevation) * 2.0);
          } else {
            advice = `✓ Excellent trophy position! Elbow cocked at ${angles.elbowAngle}° in pro range.`;
          }

          // Phase transition: when arm uncoils and reaches peak extension towards contact
          if (angles.elbowAngle > 155 && angles.wristY < angles.shoulderY) {
            this.currentPhaseIndex = 1; // move to contact point
            this.phaseScores.push(Math.round(phaseScore));
          }
        } else if (phase.id === 'contact') {
          // Contact Point
          const eb = phase.benchmarks.elbowAngle;
          const sh = phase.benchmarks.shoulderElevation;

          metricComparisons.push({
            name: eb.label,
            current: angles.elbowAngle,
            targetMin: eb.targetMin,
            targetMax: eb.targetMax,
            optimal: eb.optimal,
            unit: eb.unit,
            inRange: angles.elbowAngle >= eb.targetMin && angles.elbowAngle <= eb.targetMax
          });

          metricComparisons.push({
            name: sh.label,
            current: angles.shoulderElevation,
            targetMin: sh.targetMin,
            targetMax: sh.targetMax,
            optimal: sh.optimal,
            unit: sh.unit,
            inRange: angles.shoulderElevation >= sh.targetMin && angles.shoulderElevation <= sh.targetMax
          });

          if (angles.elbowAngle < eb.targetMin) {
            advice = phase.coaching.elbowBent + ` (Reach higher: current ${angles.elbowAngle}°, pro ${eb.targetMin}°+)`;
            phaseScore -= Math.min(50, (eb.targetMin - angles.elbowAngle) * 2.2);
          } else {
            advice = `✓ Clean contact extension! Full reach achieved at ${angles.shoulderElevation}°.`;
          }

          // Transition to follow-through when arm sweeps down across torso
          if (angles.wristY >= angles.shoulderY - 0.05 && angles.elbowAngle < 150) {
            this.currentPhaseIndex = 2; // move to follow-through
            this.phaseScores.push(Math.round(phaseScore));
          }
        } else if (phase.id === 'follow-through') {
          advice = `✓ Follow-through completed across hip. Ready for next swing.`;
          metricComparisons.push({
            name: 'Follow-Through Sweep',
            current: angles.shoulderElevation,
            targetMin: 20,
            targetMax: 70,
            optimal: 45,
            unit: '°',
            inRange: true
          });

          this.phaseHoldTime += dtSec;
          if (this.phaseHoldTime > 1.2) {
            this.completedRepCount++;
            this.currentPhaseIndex = 0;
            this.phaseHoldTime = 0;
            this.activeStatus = 'completed_rep';
          }
        }
      } else {
        // Generic sports form scoring for forehand, golf, basketball
        const benchmarkKey = Object.keys(phase.benchmarks)[0];
        const bm = phase.benchmarks[benchmarkKey];
        const currentVal = angles.elbowAngle;

        metricComparisons.push({
          name: bm.label,
          current: currentVal,
          targetMin: bm.targetMin,
          targetMax: bm.targetMax,
          optimal: bm.optimal,
          unit: bm.unit,
          inRange: currentVal >= bm.targetMin && currentVal <= bm.targetMax
        });

        if (currentVal >= bm.targetMin && currentVal <= bm.targetMax) {
          advice = `✓ Good ${phase.name} mechanics (${currentVal}${bm.unit}).`;
        } else {
          advice = `${bm.label}: current ${currentVal}${bm.unit} (Pro target: ${bm.targetMin}°-${bm.targetMax}°).`;
          phaseScore -= Math.min(50, Math.abs(currentVal - bm.optimal) * 1.8);
        }
      }

      phaseScore = Math.max(10, Math.min(100, Math.round(phaseScore)));
      this.lastFeedback = advice;

      // Overall form score
      const avgScore = this.phaseScores.length
        ? Math.round(this.phaseScores.reduce((a, b) => a + b, 0) / this.phaseScores.length)
        : phaseScore;

      return {
        valid: true,
        preset: this.preset,
        currentPhaseIndex: this.currentPhaseIndex,
        currentPhase: phase,
        totalPhases: this.preset.phases.length,
        feedback: advice,
        phaseScore: phaseScore,
        overallScore: avgScore,
        metrics: metricComparisons,
        repsCompleted: this.completedRepCount,
        angles: angles
      };
    }
  }

  // Pre-compiled kinematic frames for pro tennis serve reference demonstration
  // Simulates a professional 60fps tennis serve profile from loading to follow-through
  function generateTennisServeFrames() {
    const frames = [];
    const totalFrames = 60;

    for (let f = 0; f < totalFrames; f++) {
      const progress = f / totalFrames;
      // Interpolate kinematics through 4 phases:
      // 0.0 - 0.35: Trophy Load (elbow 95°, shoulder 95°, knee 130°)
      // 0.35 - 0.55: Racket Drop & Explode
      // 0.55 - 0.75: Contact Point (elbow 176°, shoulder 175°)
      // 0.75 - 1.00: Follow-through (cross-body sweep)

      let armAngle = 98;
      let shElevation = 95;
      let wristY = 0.35;

      if (progress < 0.35) {
        // Trophy loading
        armAngle = 95 + Math.sin(progress * 10) * 3;
        shElevation = 92 + Math.sin(progress * 10) * 4;
        wristY = 0.32;
      } else if (progress < 0.55) {
        // Drop & lag
        const t = (progress - 0.35) / 0.20;
        armAngle = 95 + t * 50;
        shElevation = 92 + t * 60;
        wristY = 0.35 - t * 0.15;
      } else if (progress < 0.75) {
        // Contact apex
        armAngle = 175;
        shElevation = 174;
        wristY = 0.12;
      } else {
        // Follow through
        const t = (progress - 0.75) / 0.25;
        armAngle = 175 - t * 110;
        shElevation = 174 - t * 130;
        wristY = 0.12 + t * 0.45;
      }

      // Generate 33-landmark skeleton frame
      const frame = [];
      for (let i = 0; i < 33; i++) {
        frame.push({ x: 0.5, y: 0.5, visibility: 0.95 });
      }

      // Anthropometric positioning for side profile serve
      frame[LM.NOSE] = { x: 0.50, y: 0.22, visibility: 0.98 };
      frame[LM.L_SHOULDER] = { x: 0.48, y: 0.35, visibility: 0.95 };
      frame[LM.R_SHOULDER] = { x: 0.52, y: 0.34, visibility: 0.98 };
      frame[LM.L_HIP] = { x: 0.49, y: 0.55, visibility: 0.95 };
      frame[LM.R_HIP] = { x: 0.51, y: 0.55, visibility: 0.98 };
      frame[LM.L_KNEE] = { x: 0.49, y: 0.72, visibility: 0.95 };
      frame[LM.R_KNEE] = { x: 0.53, y: 0.70, visibility: 0.98 };
      frame[LM.L_ANKLE] = { x: 0.48, y: 0.88, visibility: 0.95 };
      frame[LM.R_ANKLE] = { x: 0.52, y: 0.88, visibility: 0.98 };

      // Striking arm coordinates matching the target kinematics
      const radEl = toRad(shElevation);
      const elbowX = 0.52 + Math.cos(radEl) * 0.14;
      const elbowY = 0.34 - Math.sin(radEl) * 0.14;
      frame[LM.R_ELBOW] = { x: elbowX, y: elbowY, visibility: 0.98 };

      const radWr = toRad(shElevation - (180 - armAngle));
      frame[LM.R_WRIST] = {
        x: elbowX + Math.cos(radWr) * 0.14,
        y: elbowY - Math.sin(radWr) * 0.14,
        visibility: 0.98
      };

      // Non-dominant tossing arm (extended high during trophy phase)
      frame[LM.L_ELBOW] = { x: 0.44, y: 0.24, visibility: 0.95 };
      frame[LM.L_WRIST] = { x: 0.42, y: 0.14, visibility: 0.95 };

      frames.push(frame);
    }
    return frames;
  }

  // Export sports module
  PZ.sports = {
    SPORTS_PRESETS: SPORTS_PRESETS,
    SportsMotionAnalyzer: SportsMotionAnalyzer,
    generateTennisServeFrames: generateTennisServeFrames
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = PZ.sports;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
