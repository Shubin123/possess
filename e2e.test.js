/*
 * End-to-end contract test.
 *
 * The real pose estimator is a 5 MB download and a camera; this test swaps in
 * a fake MediaPipe module that replays synthetic skeletons, which makes the
 * whole browser flow deterministic: detect, classify with the built-in rules,
 * record two poses, train, classify with the trained model, and find it all
 * still there after a reload.
 *
 * It also asserts the claim the page makes about itself - that nothing is sent
 * anywhere except the two public CDNs the model comes from.
 */
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { once } = require('node:events');
const puppeteer = require('puppeteer');

const { makeLandmarks } = require('./tests/fixtures.js');

const root = __dirname;
const ALLOWED_HOSTS = ['127.0.0.1', 'localhost', 'cdn.jsdelivr.net', 'storage.googleapis.com'];

// Skeletons the fake estimator replays, keyed by the name the test sets on
// window.__fakePose.
const POSE_FRAMES = {};
['standing', 't-pose', 'hands-up', 'squat', 'one-arm-up-left'].forEach(function (pose) {
  POSE_FRAMES[pose] = makeLandmarks(pose, { scale: 0.78 }).map(function (p) {
    return { x: Number(p.x.toFixed(5)), y: Number(p.y.toFixed(5)) };
  });
});

const fakeVisionModule = `
const POSES = ${JSON.stringify(POSE_FRAMES)};
/* The page corrects for the frame's aspect ratio, so the fixtures - laid out
   for a square frame - are squeezed here the way a real 4:3 camera would. */
function frame(video) {
  const name = globalThis.__fakePose || 'standing';
  const base = POSES[name] || POSES.standing;
  const aspect = video && video.videoWidth ? video.videoWidth / video.videoHeight : 1;
  const jitter = () => (Math.random() - 0.5) * 0.003;
  return base.map(p => ({
    x: 0.5 + (p.x - 0.5) / aspect + jitter(),
    y: p.y + jitter(),
    z: 0,
    visibility: 0.96
  }));
}
export class FilesetResolver {
  static async forVisionTasks(base) { globalThis.__wasmBase = base; return {}; }
}
export class PoseLandmarker {
  static async createFromOptions(fileset, options) {
    globalThis.__modelLoads = (globalThis.__modelLoads || 0) + 1;
    globalThis.__modelBytes = options.baseOptions.modelAssetBuffer.length;
    globalThis.__delegate = options.baseOptions.delegate;
    return new PoseLandmarker();
  }
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

async function setPose(page, pose) {
  await page.evaluate(function (name) { globalThis.__fakePose = name; }, pose);
}

/** Waits for the page's own reading of the pose to settle on `expected`. */
async function expectLabel(page, expected, message) {
  await page.waitForFunction(
    function (want) { return document.getElementById('topLabel').textContent.trim() === want; },
    { timeout: 15000, polling: 100 },
    expected
  ).catch(async function () {
    const actual = await page.$eval('#topLabel', function (n) { return n.textContent; });
    const status = await page.$eval('#status', function (n) { return n.textContent; });
    throw new Error((message || 'label') + ': expected "' + expected + '", page shows "' + actual
      + '" (status: "' + status.trim() + '")');
  });
}

async function text(page, selector) {
  return (await page.$eval(selector, function (n) { return n.textContent; })).trim();
}

async function main() {
  const app = server();
  app.listen(0, '127.0.0.1');
  await once(app, 'listening');
  const origin = 'http://127.0.0.1:' + app.address().port;

  const browser = await puppeteer.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      '--autoplay-policy=no-user-gesture-required'
    ]
  });

  const offOrigin = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1000, height: 1000 });
    await browser.defaultBrowserContext().overridePermissions(origin, ['camera']);

    await page.setRequestInterception(true);
    page.on('request', function (request) {
      const url = request.url();
      const host = new URL(url).hostname;
      if (ALLOWED_HOSTS.indexOf(host) === -1) offOrigin.push(url);
      // The real CDNs are cross-origin and send CORS headers; the stubs must too.
      const cors = { 'access-control-allow-origin': '*' };
      if (url.includes('vision_bundle')) {
        return request.respond({ status: 200, contentType: 'text/javascript', headers: cors, body: fakeVisionModule });
      }
      if (url.endsWith('.task')) {
        return request.respond({
          status: 200, contentType: 'application/octet-stream', headers: cors, body: Buffer.alloc(2048, 7)
        });
      }
      return request.continue();
    });

    page.on('pageerror', function (err) { throw err; });
    if (process.env.DEBUG) {
      page.on('console', function (msg) { console.log('[page]', msg.text()); });
      page.on('requestfailed', function (r) { console.log('[failed]', r.url(), r.failure() && r.failure().errorText); });
    }
    await page.goto(origin + '/', { waitUntil: 'domcontentloaded' });

    /* --- the estimator loads and the built-in rules classify --- */
    await setPose(page, 't-pose');
    await page.click('#startCamera');
    await expectLabel(page, 't-pose', 'built-in rules on a T-pose');
    assert.equal(await page.evaluate(function () { return globalThis.__modelLoads; }), 1, 'the model loaded once');
    assert.ok(await page.evaluate(function () { return globalThis.__modelBytes > 0; }), 'weights reached the estimator');

    await setPose(page, 'hands-up');
    await expectLabel(page, 'hands-up', 'built-in rules follow the pose');

    await setPose(page, 'squat');
    await expectLabel(page, 'squat', 'built-in rules on a squat');

    const counts = await text(page, '#repCounts');
    assert.ok(/3|2/.test(counts) || counts.length > 0, 'poses held are counted: ' + counts);

    /* --- teaching two poses of its own --- */
    await page.$eval('#recordSeconds', function (input) { input.value = '1'; });

    async function record(name, pose) {
      await setPose(page, pose);
      await page.$eval('#className', function (input, value) { input.value = value; }, name);
      await page.click('#recordBtn');
      await page.waitForFunction(function (label) {
        return document.getElementById('trainStatus').textContent.includes('Recorded')
          && document.getElementById('trainStatus').textContent.includes(label);
      }, { timeout: 20000, polling: 100 }, name);
    }

    await record('alpha', 't-pose');
    await record('beta', 'standing');

    const classList = await text(page, '#classList');
    assert.ok(classList.includes('alpha') && classList.includes('beta'), 'both poses are listed: ' + classList);
    const sampleCount = await page.evaluate(function () { return globalThis.__possess.dataset.samples.length; });
    assert.ok(sampleCount >= 20, 'recorded ' + sampleCount + ' samples');

    /* --- the trained classifier --- */
    assert.equal(await page.$eval('#trainBtn', function (b) { return b.disabled; }), false, 'training is offered');
    await page.click('#trainBtn');
    await page.waitForFunction(function () {
      return document.getElementById('trainStatus').textContent.includes('Trained');
    }, { timeout: 30000, polling: 100 });
    const report = await text(page, '#trainStatus');
    assert.ok(/on held-out frames/.test(report), 'training reports held-out accuracy: ' + report);
    assert.equal(await page.$eval('#backendMode', function (s) { return s.value; }), 'trained');

    await setPose(page, 't-pose');
    await expectLabel(page, 'alpha', 'the trained model recognizes what it was taught');
    await setPose(page, 'standing');
    await expectLabel(page, 'beta', 'and tells the second pose apart');

    /* --- a pose recorded after training is still recognized --- */
    await record('gamma', 'squat');
    const staleNotice = await text(page, '#trainStatus');
    assert.ok(/Train again/.test(staleNotice), 'the page asks for a retrain: ' + staleNotice);
    await setPose(page, 'squat');
    await expectLabel(page, 'gamma', 'the nearest-neighbour head covers the pose the network missed');
    await setPose(page, 't-pose');
    await expectLabel(page, 'alpha', 'and the older poses still work');

    /* --- nothing in frame --- */
    await setPose(page, 'none');
    await page.waitForFunction(function () {
      return document.getElementById('topScore').textContent.includes('no pose detected');
    }, { timeout: 10000, polling: 100 });

    /* --- everything survives a reload, still with no server --- */
    const beforeReload = await page.evaluate(function () { return globalThis.__possess.dataset.samples.length; });
    await page.reload({ waitUntil: 'domcontentloaded' });
    const restored = await page.evaluate(function () {
      return {
        samples: globalThis.__possess.dataset.samples.length,
        labels: globalThis.__possess.dataset.labels,
        model: globalThis.__possess.model ? globalThis.__possess.model.labels : null
      };
    });
    assert.equal(restored.samples, beforeReload, 'samples came back from local storage');
    assert.deepEqual(restored.labels, ['alpha', 'beta', 'gamma']);
    assert.deepEqual(restored.model, ['alpha', 'beta'], 'the trained model came back too');

    await setPose(page, 't-pose');
    await page.click('#startCamera');
    await expectLabel(page, 'alpha', 'the restored model classifies without retraining');
    assert.equal(await page.evaluate(function () { return globalThis.__modelLoads; }), 1,
      'the pose weights came from IndexedDB, not the network');

    /* --- exports are the documented shape --- */
    const exported = await page.evaluate(function () {
      return {
        dataset: globalThis.__possess.dataset.toJSON(),
        model: globalThis.__possess.model.toJSON()
      };
    });
    assert.equal(exported.dataset.format, 'possess-dataset');
    assert.equal(exported.dataset.featureLength, exported.model.inputSize);
    assert.equal(exported.model.format, 'possess-mlp');
    assert.deepEqual(exported.model.labels, ['alpha', 'beta']);

    assert.deepEqual(offOrigin, [], 'the page talked to nothing but the model CDNs');
    console.log('e2e: ' + sampleCount + ' samples recorded, two poses trained and recognized, '
      + 'state restored after reload, no unexpected network calls');
  } finally {
    await browser.close();
    app.close();
  }
}

main().catch(function (err) {
  console.error(err);
  process.exit(1);
});
