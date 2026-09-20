const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { once } = require('node:events');
const puppeteer = require('puppeteer');
const { makeLandmarks } = require('./fixtures.js');

const root = path.resolve(__dirname, '..');

const POSE_FRAMES = {};
['standing', 't-pose', 'hands-up', 'squat', 'one-arm-up-left'].forEach(function (pose) {
  POSE_FRAMES[pose] = makeLandmarks(pose, { scale: 0.78 }).map(function (p) {
    return { x: Number(p.x.toFixed(5)), y: Number(p.y.toFixed(5)) };
  });
});

const fakeVisionModule = `
const POSES = ${JSON.stringify(POSE_FRAMES)};
function frame(video) {
  const name = globalThis.__fakePose || 'standing';
  const base = POSES[name] || POSES.standing;
  return base.map(p => ({ x: p.x, y: p.y, visibility: 0.95 }));
}
export class FilesetResolver {
  static async forVisionTasks() { return {}; }
}
export class PoseLandmarker {
  static async createFromOptions() { return new PoseLandmarker(); }
  async setOptions() {}
  detectForVideo(video) {
    if (globalThis.__fakePose === 'none') return { landmarks: [], worldLandmarks: [] };
    return { landmarks: [frame(video)], worldLandmarks: [] };
  }
  detect(source) { return this.detectForVideo(source); }
  close() {}
}
`;

function server() {
  return http.createServer(function (req, res) {
    const requested = req.url === '/' ? '/index.html' : req.url;
    const file = path.resolve(root, '.' + requested.split('?')[0]);
    if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); return res.end();
    }
    const type = file.endsWith('.html') ? 'text/html'
      : file.endsWith('.js') ? 'text/javascript'
      : file.endsWith('.css') ? 'text/css'
      : file.endsWith('.svg') ? 'image/svg+xml' : 'application/octet-stream';
    res.writeHead(200, { 'content-type': type });
    res.end(fs.readFileSync(file));
  });
}

async function testHealthDashboard() {
  const app = server();
  app.listen(0, '127.0.0.1');
  await once(app, 'listening');
  const origin = 'http://127.0.0.1:' + app.address().port;

  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      '--autoplay-policy=no-user-gesture-required'
    ]
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1400, height: 1100 });
    await browser.defaultBrowserContext().overridePermissions(origin, ['camera']);

    await page.setRequestInterception(true);
    page.on('request', function (request) {
      const url = request.url();
      const cors = { 'access-control-allow-origin': '*' };
      if (url.includes('vision_bundle')) {
        return request.respond({ status: 200, contentType: 'text/javascript', headers: cors, body: fakeVisionModule });
      }
      if (url.endsWith('.task')) {
        return request.respond({ status: 200, contentType: 'application/octet-stream', headers: cors, body: Buffer.alloc(2048, 7) });
      }
      return request.continue();
    });

    await page.goto(origin + '/', { waitUntil: 'domcontentloaded' });

    // Verify all 7 widgets are present in the DOM
    const widgetIds = await page.$$eval('.widget', ws => ws.map(w => w.id));
    assert.deepEqual(widgetIds, [
      'widgetSource', 'widgetPrediction', 'widgetTeach', 'widgetSports',
      'widgetPosture', 'widgetRoutine', 'widgetRom'
    ], 'All 7 modules must be in the dashboard');

    // Start camera and feed standing pose
    await page.evaluate(() => { globalThis.__fakePose = 'standing'; });
    await page.click('#startCamera');

    // Wait for posture score to update
    await page.waitForFunction(() => {
      const score = document.getElementById('postureScoreNum');
      return score && score.textContent !== '—' && Number(score.textContent) >= 80;
    }, { timeout: 6000 });

    const postureText = await page.$eval('#postureLabel', el => el.textContent);
    assert.ok(/Good posture/.test(postureText), 'Standing pose should be rated good posture');

    // Test Calibrate button
    await page.click('#calibratePosture');
    await page.waitForFunction(() => document.getElementById('calibratePosture').textContent.includes('Calibrated'));

    // Test Guided Routine: start desk-reset routine
    await page.click('#routineToggleBtn');
    await page.waitForFunction(() => {
      const step = document.getElementById('routineStepName');
      return step && step.textContent.includes('Upright');
    });

    // Hold standing pose and verify progress advances
    await page.waitForFunction(() => {
      const timer = document.getElementById('routineHoldTimer');
      return timer && parseFloat(timer.textContent) > 0.5;
    }, { timeout: 6000 });

    // Test Range of Motion goniometer
    // Switch to hands-up pose
    await page.evaluate(() => { globalThis.__fakePose = 'hands-up'; });
    await page.waitForFunction(() => {
      const deg = document.getElementById('romCurrentDeg');
      return deg && parseInt(deg.textContent, 10) >= 120;
    }, { timeout: 6000 });

    // Start 5s ROM mobility test
    await page.click('#romTestBtn');
    const testingBtnText = await page.$eval('#romTestBtn', el => el.textContent);
    assert.ok(testingBtnText.includes('Testing'), 'Test button should indicate testing');

    // Test Sports Motion Trainer pro demo
    await page.click('#loadTennisDemoBtn');
    await page.waitForFunction(() => {
      const score = document.getElementById('sportsScoreNum');
      return score && score.textContent !== '--' && parseInt(score.textContent, 10) >= 80;
    }, { timeout: 6000 });

    const guidance = await page.$eval('#sportsCameraGuidanceText', el => el.textContent);
    assert.ok(guidance.includes('Side Profile'), 'Guidance should recommend side profile: ' + guidance);

    if (process.env.SCREENSHOT) {
      await page.screenshot({ path: process.env.SCREENSHOT, fullPage: true });
    }

    console.log('✔ All 7 dashboard modules passed end-to-end browser integration verification!');
  } finally {
    await browser.close();
    app.close();
  }
}

testHealthDashboard().catch(err => {
  console.error(err);
  process.exit(1);
});
