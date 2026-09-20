const test = require('node:test');
const assert = require('node:assert/strict');
const sports = require('../sports.js');
const { makeLandmarks } = require('./fixtures.js');

test('SportsMotionAnalyzer initializes with tennis serve and camera guidance', () => {
  const analyzer = new sports.SportsMotionAnalyzer();
  assert.equal(analyzer.preset.id, 'tennis-serve');
  assert.ok(analyzer.preset.cameraGuidance.includes('Profile'));
  assert.equal(analyzer.preset.phases.length, 3);
});

test('SportsMotionAnalyzer gives positive feedback on optimal tennis trophy pose', () => {
  const analyzer = new sports.SportsMotionAnalyzer();
  analyzer.setPreset('tennis-serve');

  // Generate tennis serve synthetic frames
  const frames = sports.generateTennisServeFrames();
  const trophyFrame = frames[5]; // Early frame in trophy pose

  const result = analyzer.analyze(trophyFrame);
  assert.equal(result.valid, true);
  assert.equal(result.currentPhase.id, 'trophy');
  assert.ok(result.phaseScore >= 80, 'Optimal trophy pose should score >= 80: ' + result.phaseScore);
  assert.ok(result.feedback.includes('trophy') || result.feedback.includes('Excellent'), result.feedback);
});

test('SportsMotionAnalyzer detects elbow form fault in trophy pose and gives coaching cue', () => {
  const analyzer = new sports.SportsMotionAnalyzer();
  analyzer.setPreset('tennis-serve');

  const frames = sports.generateTennisServeFrames();
  // Modify frame so elbow is collapsed/too closed (< 70°)
  const faultyFrame = frames[5].map(p => ({ ...p }));
  // Move wrist closer to shoulder
  faultyFrame[16] = { x: faultyFrame[14].x - 0.05, y: faultyFrame[14].y + 0.05, visibility: 0.98 };

  const result = analyzer.analyze(faultyFrame);
  assert.equal(result.valid, true);
  assert.ok(result.feedback.includes('elbow') || result.feedback.includes('Open'), 'Feedback should advise opening elbow: ' + result.feedback);
  assert.ok(result.phaseScore < 85, 'Faulty form should lower phase score');
});

test('SportsMotionAnalyzer tracks entire tennis serve sequence through all phases', () => {
  const analyzer = new sports.SportsMotionAnalyzer();
  analyzer.setPreset('tennis-serve');

  const frames = sports.generateTennisServeFrames();
  assert.equal(frames.length, 60);

  const phaseSequence = [];
  for (let i = 0; i < frames.length; i++) {
    const res = analyzer.analyze(frames[i], 0.033);
    if (!phaseSequence.includes(res.currentPhase.id)) {
      phaseSequence.push(res.currentPhase.id);
    }
  }

  assert.ok(phaseSequence.includes('trophy'), 'Must pass through trophy phase');
  assert.ok(phaseSequence.includes('contact'), 'Must pass through contact phase');
  assert.ok(phaseSequence.includes('follow-through'), 'Must pass through follow-through phase');
});

test('Sports presets include Golf, Basketball, and Tennis Forehand', () => {
  const analyzer = new sports.SportsMotionAnalyzer();

  analyzer.setPreset('tennis-forehand');
  assert.equal(analyzer.preset.sport, 'Tennis');
  assert.equal(analyzer.preset.phases[0].id, 'unit-turn');

  analyzer.setPreset('golf-swing');
  assert.equal(analyzer.preset.sport, 'Golf');
  assert.ok(analyzer.preset.cameraGuidance.includes('Down-The-Line'));

  analyzer.setPreset('basketball-shot');
  assert.equal(analyzer.preset.sport, 'Basketball');
  assert.equal(analyzer.preset.phases[0].id, 'set-point');
});
