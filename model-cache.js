/*
 * poser - persistent browser storage for the pose model file.
 *
 * The Pose Landmarker weights are 5-30 MB depending on the variant. They are
 * stored as raw ArrayBuffers in IndexedDB so the download happens once and
 * every later visit starts offline-fast. IndexedDB rather than the Cache API:
 * the model is served from a Google Storage bucket through redirects, which
 * the Cache API refuses to store from a manually built Response.
 */
(function (global) {
  const PZ = (global.PZ = global.PZ || {});

  const DB_NAME = 'poser-model-cache';
  const DB_VERSION = 1;
  const STORE = 'models';
  let _db = null;

  function openDB() {
    if (_db) return Promise.resolve(_db);
    if (!global.indexedDB) return Promise.reject(new Error('IndexedDB unavailable'));
    return new Promise(function (resolve, reject) {
      const req = global.indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = function () {
        if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
      };
      req.onsuccess = function () { _db = req.result; resolve(_db); };
      req.onerror = function () { reject(req.error); };
    });
  }

  function withStore(mode, run) {
    return openDB().then(function (db) {
      return new Promise(function (resolve, reject) {
        const tx = db.transaction(STORE, mode);
        const req = run(tx.objectStore(STORE));
        req.onsuccess = function () { resolve(req.result); };
        req.onerror = function () { reject(req.error); };
      });
    });
  }

  /** Size in bytes of the cached model, or null when it is not stored. */
  async function cachedSize(url) {
    try {
      const value = await withStore('readonly', function (store) { return store.get(url); });
      return value ? value.byteLength : null;
    } catch (err) { return null; }
  }

  async function store(url, buffer) {
    try {
      await withStore('readwrite', function (s) { return s.put(buffer, url); });
      return true;
    } catch (err) { return false; }
  }

  async function remove(url) {
    try {
      await withStore('readwrite', function (s) { return s.delete(url); });
      return true;
    } catch (err) { return false; }
  }

  /*
   * Returns the model bytes, from IndexedDB when present and from the network
   * otherwise, reporting {loaded, total, cached} to `onProgress` as it goes.
   */
  async function fetchModel(url, onProgress) {
    try {
      const hit = await withStore('readonly', function (s) { return s.get(url); });
      if (hit) {
        if (onProgress) onProgress({ loaded: hit.byteLength, total: hit.byteLength, cached: true, done: true });
        return new Uint8Array(hit);
      }
    } catch (err) { /* fall through to the network */ }

    const response = await fetch(url);
    if (!response.ok) throw new Error('Could not download the model (HTTP ' + response.status + ')');
    const total = Number(response.headers.get('content-length')) || 0;

    let buffer;
    if (response.body && response.body.getReader) {
      const reader = response.body.getReader();
      const chunks = [];
      let loaded = 0;
      for (;;) {
        const chunk = await reader.read();
        if (chunk.done) break;
        chunks.push(chunk.value);
        loaded += chunk.value.length;
        if (onProgress) onProgress({ loaded: loaded, total: total, cached: false, done: false });
      }
      const merged = new Uint8Array(loaded);
      let at = 0;
      chunks.forEach(function (c) { merged.set(c, at); at += c.length; });
      buffer = merged.buffer;
    } else {
      buffer = await response.arrayBuffer();
    }

    await store(url, buffer);
    if (onProgress) onProgress({ loaded: buffer.byteLength, total: buffer.byteLength, cached: false, done: true });
    return new Uint8Array(buffer);
  }

  PZ.modelCache = { fetchModel: fetchModel, cachedSize: cachedSize, store: store, remove: remove };
})(typeof globalThis !== 'undefined' ? globalThis : this);
