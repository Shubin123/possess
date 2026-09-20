const test = require('node:test');
const assert = require('node:assert/strict');
const { makeLandmarks } = require('./fixtures.js');
const health = require('../health.js');

test('PostureMonitor scores upright standing pose near 100', () => {
  const pm = new health.PostureMonitor();
  const upright = makeLandmarks('standing');
  const res = pm.analyze(upright);
  assert.equal(res.valid, true);
  assert.ok(res.score >= 90, 'Upright standing should score >= 90: ' + res.score);
  assert.equal(res.status, 'good');
});

test('PostureMonitor penalizes forward head posture', () => {
  const pm = new health.PostureMonitor();
  const upright = makeLandmarks('standing');
  const initial = pm.analyze(upright);

  // Shift ears forward (in MediaPipe frame, x moves forward)
  const forwardHead = upright.map(p => ({ ...p }));
  forwardHead[7] = { ...forwardHead[7], x: forwardHead[7].x + 0.12 };
  forwardHead[8] = { ...forwardHead[8], x: forwardHead[8].x + 0.12 };

  const slouched = pm.analyze(forwardHead);
  assert.ok(slouched.metrics.headAngle > initial.metrics.headAngle, 'Head angle should increase');
  assert.ok(slouched.score < initial.score, 'Forward head should lower the score');
});

test('PostureMonitor detects lateral shoulder asymmetry', () => {
  const pm = new health.PostureMonitor();
  const upright = makeLandmarks('standing');
  const uneven = upright.map(p => ({ ...p }));

  // Drop left shoulder down by 0.08
  uneven[11] = { ...uneven[11], y: uneven[11].y + 0.08 };

  const res = pm.analyze(uneven);
  assert.ok(res.metrics.shoulderTilt >= 4, 'Shoulder tilt should be >= 4 degrees: ' + res.metrics.shoulderTilt);
  assert.ok(res.score < 90, 'Uneven shoulders should lower the posture score');
});

test('PostureMonitor calibrates upright baseline', () => {
  const pm = new health.PostureMonitor();
  const upright = makeLandmarks('standing');
  const base = pm.calibrateBaseline(upright);
  assert.ok(base.torsoLength > 0, 'Torso length should be calibrated');
  assert.ok(base.shoulderWidth > 0, 'Shoulder width should be calibrated');

  const res = pm.analyze(upright);
  assert.equal(res.hasBaseline, true);
  assert.ok(res.metrics.torsoCompression >= 95, 'Should be near 100% upright');
});

test('RoutineEngine steps through routine, holds pose, and finishes', () => {
  const engine = new health.RoutineEngine(false); // audio disabled for test
  const status0 = engine.start('desk-reset');
  assert.equal(status0.state, 'holding');
  assert.equal(status0.routineId, 'desk-reset');
  assert.equal(status0.stepIndex, 0);
  assert.equal(status0.currentRep, 1);

  const step0 = engine.getCurrentStep();
  const requiredPose = step0.pose;

  // Simulate holding the pose for full duration
  let s = status0;
  for (let i = 0; i < 50; i++) {
    s = engine.update(requiredPose, 0.95, 0.1);
  }
  assert.ok(s.currentRep >= 2 || s.stepIndex >= 1, 'Holding pose should advance reps');
});

test('RomGoniometer computes joint angles and grades mobility', () => {
  const rom = new health.RomGoniometer();
  const standing = makeLandmarks('standing');
  const tpose = makeLandmarks('t-pose');
  const handsup = makeLandmarks('hands-up');

  // Standing arms down: shoulder angle ~ 10°-30°
  const shStanding = rom.calculateJointAngle('shoulder_r', standing);
  // T-pose arms out: shoulder angle ~ 85°-95°
  const shTpose = rom.calculateJointAngle('shoulder_r', tpose);
  // Hands-up arms elevated: shoulder angle ~ 160°-180°
  const shHandsUp = rom.calculateJointAngle('shoulder_r', handsup);

  assert.ok(shStanding < shTpose, 'Standing angle should be less than T-pose');
  assert.ok(shTpose < shHandsUp, 'T-pose angle should be less than Hands Up');
  assert.ok(shHandsUp >= 150, 'Hands-up should measure high elevation: ' + shHandsUp);

  // Test assessment mode
  rom.setJoint('shoulder_r');
  rom.startTest(0.2); // 0.2s duration for fast test
  const res1 = rom.update(handsup, 0.1);
  assert.equal(res1.testActive, true);
  const res2 = rom.update(handsup, 0.15); // Exceeds duration
  assert.equal(res2.testActive, false);
  assert.ok(res2.completedTestResult, 'Test result should be generated');
  assert.ok(res2.completedTestResult.percentNormal >= 80, 'Hands-up should achieve high % of normal');
  assert.ok(/Normal|Mild/.test(res2.completedTestResult.grade));
});

test('HealthSessionManager exports structured session data', () => {
  const pm = new health.PostureMonitor();
  const engine = new health.RoutineEngine(false);
  const rom = new health.RomGoniometer();
  const sm = new health.HealthSessionManager();

  const report = sm.generateReport(pm, engine, rom);
  assert.equal(report.appName, 'possess');
  assert.ok(report.postureAnalysis);
  assert.ok(report.therapyRoutines);
  assert.ok(report.rangeOfMotion);

  const jsonStr = sm.exportJSON(pm, engine, rom);
  const parsed = JSON.parse(jsonStr);
  assert.equal(parsed.appName, 'possess');
});
