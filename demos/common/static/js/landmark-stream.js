/**
 * Browser-side MediaPipe -> WebSocket landmark stream.
 *
 * Vision runs here in the browser; Flask owns the gesture logic. Each frame we
 * run the requested tasks-vision landmarkers, normalise the output into the same
 * shape demos/common/landmarks.py expects, and push it over a persistent socket.
 * The server answers with state JSON which the page draws.
 *
 * Two guards keep a slow link from turning into lag:
 *   - client side: a frame is skipped when the socket still has buffered bytes,
 *     so we never queue stale landmarks.
 *   - server side: a monotonic `seq` lets FrameGuard drop anything out of order.
 *
 * Mirroring: getUserMedia gives an unmirrored feed and we mirror it for display
 * (x -> 1-x), which is the selfie orientation MediaPipe's handedness classifier
 * already assumes. So the label it returns matches the hand you see on screen and
 * must NOT be flipped again — doing so is what made left/right controls come out
 * backwards. Set `swapHandedness: true` (or the demo's config flag) if a
 * particular camera still reports them inverted.
 */

const TASKS_VERSION = '0.10.21';
const CDN_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
const MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models';
/** Locally vendored copies, if `demos/tools/vendor_web_assets.py` has been run. */
const VENDOR_BASE = '/shared/static/vendor';

const CDN_MODELS = {
  hands: `${MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
  face: `${MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
  pose: `${MODEL_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
};
const VENDOR_MODELS = {
  hands: `${VENDOR_BASE}/hand_landmarker.task`,
  face: `${VENDOR_BASE}/face_landmarker.task`,
  pose: `${VENDOR_BASE}/pose_landmarker_lite.task`,
};

/** Approximate download sizes, shown while loading so a slow link is legible. */
const MODEL_MB = { hands: 7.8, face: 3.8, pose: 5.5 };
const WASM_MB = 7.3;

/** After this long without finishing a stage, explain rather than look hung. */
const SLOW_LOAD_MS = 6000;
/** Hard ceiling on the whole setup, so a wedged fetch fails loudly. */
const LOAD_TIMEOUT_MS = 180000;

/** Reject if a promise has not settled in time, naming the stage that stalled. */
function withTimeout(promise, ms, label) {
  let timer;
  const guard = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`)),
      ms,
    );
  });
  return Promise.race([promise, guard]).finally(() => clearTimeout(timer));
}

/** True if a URL is fetchable, used to detect vendored assets. */
async function exists(url) {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return response.ok;
  } catch (err) {
    return false;
  }
}

/** Bytes still queued on the socket before we start skipping frames. */
const MAX_BUFFERED_BYTES = 48 * 1024;
const RECONNECT_DELAYS_MS = [400, 800, 1600, 3000, 5000];

function flipLabel(label) {
  if (label === 'Left') return 'Right';
  if (label === 'Right') return 'Left';
  return label || 'Unknown';
}

/** Trim coordinates to 4 decimals: sub-pixel precision we cannot use anyway. */
function round4(value) {
  return Math.round(value * 10000) / 10000;
}

export class LandmarkStream {
  /**
   * @param {object} options
   * @param {HTMLVideoElement} options.video   video element to attach the camera to
   * @param {string} options.basePath          blueprint prefix, e.g. "/air-canvas"
   * @param {object} [options.need]            {hands, face, pose} streams to run
   * @param {number} [options.numHands]        max hands to track (default 2)
   * @param {boolean} [options.mirror]         mirror display and landmarks (default true)
   * @param {function} [options.onState]       called with each server state object
   * @param {function} [options.onStatus]      called whenever connection stats change
   * @param {function} [options.onError]       called with fatal setup errors
   */
  constructor(options) {
    this.video = options.video;
    this.basePath = (options.basePath || '').replace(/\/$/, '');
    this.need = Object.assign({ hands: true, face: false, pose: false }, options.need || {});
    this.numHands = options.numHands ?? 2;
    this.mirror = options.mirror !== false;
    // Escape hatch for cameras that report handedness inverted anyway.
    this.swapHandedness = options.swapHandedness === true;
    this.onState = options.onState || (() => {});
    this.onStatus = options.onStatus || (() => {});
    this.onError = options.onError || ((err) => console.error(err));

    this.sid = (crypto.randomUUID && crypto.randomUUID()) ||
      `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`;

    this.seq = 0;
    this.sent = 0;
    this.clientDropped = 0;
    this.serverStats = null;
    this.running = false;
    this.connected = false;
    this.reconnectAttempt = 0;
    this.lastTimestamp = -1;
    this.detectFps = 0;
    this.lastDetectAt = 0;
    this.landmarkers = {};
    this.ws = null;

    // Loader state, surfaced through onStatus so the page can show progress
    // instead of one static "loading" message for the whole download.
    this.stage = 'idle';
    this.error = null;
    this.assetBase = CDN_BASE;
    this.models = CDN_MODELS;
    this.vendored = false;
    this.delegate = 'GPU';
    this._slowTimer = null;
  }

  /** Update the loader stage and notify the page. */
  setStage(stage, detail = '') {
    this.stage = stage;
    this.onStatus(this.status(detail ? `${stage} ${detail}` : stage));
  }

  /** Warn after SLOW_LOAD_MS that a stage is a large first-time download. */
  watchSlow(message) {
    clearTimeout(this._slowTimer);
    this._slowTimer = setTimeout(() => {
      this.onStatus(this.status(message));
    }, SLOW_LOAD_MS);
  }

  clearSlow() {
    clearTimeout(this._slowTimer);
    this._slowTimer = null;
  }

  /** Load models, open the camera, connect, and start detecting. */
  async start() {
    try {
      await withTimeout(this.setup(), LOAD_TIMEOUT_MS, 'setup');
    } catch (err) {
      // Record the failure so the pill turns red instead of sitting on the last
      // "loading" message forever, then rethrow for the page's catch handler.
      this.clearSlow();
      this.error = err.message;
      this.setStage('failed');
      throw err;
    }
    return this;
  }

  async setup() {
    // Prefer locally vendored assets when present: the CDN path pulls ~15 MB on
    // a cold cache, which is the difference between "instant" and "did it hang?".
    this.setStage('checking assets');
    // A HEAD probe is cheap and avoids needing the server to tell us.

    if (await exists(`${VENDOR_BASE}/vision_bundle.mjs`)) {
      this.assetBase = VENDOR_BASE;
      this.models = VENDOR_MODELS;
      this.vendored = true;
    }

    this.setStage('loading runtime');
    this.watchSlow(
      this.vendored
        ? 'initialising the MediaPipe runtime from local assets'
        : `downloading the MediaPipe runtime (~${WASM_MB} MB, first load only)`,
    );
    const vision = await import(`${this.assetBase}/vision_bundle.mjs`);
    const fileset = await vision.FilesetResolver.forVisionTasks(`${this.assetBase}/wasm`);
    this.clearSlow();

    // Sequential, not parallel: one model at a time keeps the progress message
    // honest and avoids three large downloads competing for bandwidth.
    const wanted = [
      ['hands', this.need.hands, vision.HandLandmarker, { numHands: this.numHands }],
      ['face', this.need.face, vision.FaceLandmarker, { numFaces: 1, outputFaceBlendshapes: false }],
      ['pose', this.need.pose, vision.PoseLandmarker, { numPoses: 1 }],
    ];
    for (const [name, enabled, Landmarker, extra] of wanted) {
      if (!enabled) continue;
      this.setStage('loading model', `${name} (~${MODEL_MB[name]} MB)`);
      this.watchSlow(
        this.vendored
          ? `loading the ${name} model from local assets`
          : `still downloading the ${name} model (~${MODEL_MB[name]} MB, first load only)`,
      );
      this.landmarkers[name] = await this.createLandmarker(Landmarker, fileset, name, extra);
      this.clearSlow();
    }

    this.setStage('opening camera');
    const media = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    this.video.srcObject = media;
    this.video.style.transform = this.mirror ? 'scaleX(-1)' : '';
    await this.video.play();

    this.setStage('connecting');
    this.connect();
    this.running = true;
    this.loop();
  }

  /**
   * Create one landmarker, falling back from the GPU delegate to CPU.
   *
   * Browsers with hardware acceleration disabled either fail or fall back to
   * deprecated software WebGL, so retrying on CPU is what keeps the demo working
   * instead of stalling on a delegate that will never initialise.
   */
  async createLandmarker(Landmarker, fileset, name, extra) {
    for (const delegate of ['GPU', 'CPU']) {
      try {
        const landmarker = await Landmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: this.models[name], delegate },
          runningMode: 'VIDEO',
          ...extra,
        });
        this.delegate = delegate;
        return landmarker;
      } catch (err) {
        if (delegate === 'CPU') throw err;
        console.warn(`${name}: GPU delegate failed (${err.message}), retrying on CPU`);
        this.setStage('loading model', `${name} on CPU`);
      }
    }
    throw new Error(`could not create the ${name} landmarker`);
  }

  stop() {
    this.running = false;
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    const tracks = this.video.srcObject?.getTracks?.() || [];
    tracks.forEach((track) => track.stop());
    Object.values(this.landmarkers).forEach((l) => l.close?.());
  }

  // -- socket ---------------------------------------------------------------
  wsUrl() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    return `${scheme}://${location.host}${this.basePath}/ws?sid=${this.sid}`;
  }

  connect() {
    const ws = new WebSocket(this.wsUrl());
    this.ws = ws;

    ws.onopen = () => {
      this.connected = true;
      this.reconnectAttempt = 0;
      this.onStatus(this.status('connected'));
    };
    ws.onmessage = (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch (err) {
        return;
      }
      if (payload.ack) return;           // command acknowledgement
      if (payload._meta) this.serverStats = payload._meta;
      if (!payload._meta?.skipped) this.onState(payload);
    };
    ws.onclose = () => {
      this.connected = false;
      this.onStatus(this.status('disconnected'));
      if (!this.running) return;
      const delay = RECONNECT_DELAYS_MS[
        Math.min(this.reconnectAttempt, RECONNECT_DELAYS_MS.length - 1)
      ];
      this.reconnectAttempt += 1;
      setTimeout(() => { if (this.running) this.connect(); }, delay);
    };
    ws.onerror = () => { /* onclose handles recovery */ };
  }

  /** Send a UI command (clear, undo, next slide, ...) over the socket or HTTP. */
  async sendCommand(command, payload = {}) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'command', command, payload }));
      return { queued: true };
    }
    const response = await fetch(`${this.basePath}/command`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Demo-Session': this.sid },
      body: JSON.stringify({ command, payload }),
    });
    return response.json();
  }

  /** Flip Left/Right labels from here on; returns the new setting. */
  toggleHandedness() {
    this.swapHandedness = !this.swapHandedness;
    return this.swapHandedness;
  }

  snapshotUrl() {
    return `${this.basePath}/snapshot?sid=${this.sid}`;
  }

  status(message) {
    return {
      message: message || (this.connected ? 'connected' : 'disconnected'),
      connected: this.connected,
      sent: this.sent,
      clientDropped: this.clientDropped,
      server: this.serverStats,
      detectFps: this.detectFps,
      sid: this.sid,
      stage: this.stage,
      error: this.error,
      delegate: this.delegate,
      vendored: this.vendored,
      loading: !this.running && this.stage !== 'failed',
    };
  }

  // -- detection loop -------------------------------------------------------
  loop() {
    const step = () => {
      if (!this.running) return;
      this.detectOnce();
      if (this.video.requestVideoFrameCallback) {
        this.video.requestVideoFrameCallback(step);
      } else {
        requestAnimationFrame(step);
      }
    };
    step();
  }

  detectOnce() {
    const video = this.video;
    if (!video.videoWidth || video.readyState < 2) return;

    // tasks-vision requires strictly increasing timestamps.
    let timestamp = performance.now();
    if (timestamp <= this.lastTimestamp) timestamp = this.lastTimestamp + 1;
    this.lastTimestamp = timestamp;

    const payload = {
      seq: this.seq + 1,
      ts: timestamp / 1000,
      width: video.videoWidth,
      height: video.videoHeight,
    };

    try {
      if (this.landmarkers.hands) payload.hands = this.readHands(video, timestamp);
      if (this.landmarkers.face) payload.face = this.readFace(video, timestamp);
      if (this.landmarkers.pose) payload.pose = this.readPose(video, timestamp);
    } catch (err) {
      this.onError(err);
      return;
    }

    const now = performance.now();
    if (this.lastDetectAt) {
      const instant = 1000 / Math.max(now - this.lastDetectAt, 1);
      this.detectFps = this.detectFps ? this.detectFps * 0.85 + instant * 0.15 : instant;
    }
    this.lastDetectAt = now;

    this.send(payload);
  }

  send(payload) {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      this.clientDropped += 1;
      return;
    }
    if (ws.bufferedAmount > MAX_BUFFERED_BYTES) {
      // Socket is backed up: skip this frame rather than queue stale landmarks.
      this.clientDropped += 1;
      this.onStatus(this.status('throttling'));
      return;
    }
    this.seq = payload.seq;
    ws.send(JSON.stringify(payload));
    this.sent += 1;
    if (this.sent % 15 === 0) this.onStatus(this.status());
  }

  // -- landmark extraction --------------------------------------------------
  point(landmark) {
    const x = this.mirror ? 1 - landmark.x : landmark.x;
    return { x: round4(x), y: round4(landmark.y) };
  }

  readHands(video, timestamp) {
    const result = this.landmarkers.hands.detectForVideo(video, timestamp);
    const hands = [];
    const sets = result.landmarks || [];
    for (let i = 0; i < sets.length; i += 1) {
      const category = (result.handedness || [])[i]?.[0];
      let label = category?.categoryName || 'Unknown';
      if (this.swapHandedness) label = flipLabel(label);
      hands.push({
        label,
        score: round4(category?.score ?? 0),
        points: sets[i].map((p) => this.point(p)),
      });
    }
    return hands;
  }

  readFace(video, timestamp) {
    const result = this.landmarkers.face.detectForVideo(video, timestamp);
    const set = (result.faceLandmarks || [])[0];
    if (!set) return null;
    return { points: set.map((p) => this.point(p)) };
  }

  readPose(video, timestamp) {
    const result = this.landmarkers.pose.detectForVideo(video, timestamp);
    const set = (result.landmarks || [])[0];
    if (!set) return null;
    return {
      points: set.map((p) => {
        const point = this.point(p);
        point.visibility = round4(p.visibility ?? 1);
        return point;
      }),
    };
  }
}

/** Wire a status object into the header pill and an optional status element. */
export function bindStatusUi(pillId, statusId) {
  const pill = document.getElementById(pillId);
  const box = statusId ? document.getElementById(statusId) : null;
  return (status) => {
    if (pill) {
      pill.textContent = status.error ? 'error' : status.message;
      pill.className = `pill ${status.error ? 'off' : status.connected ? 'on' : 'warn'}`;
    }
    if (!box) return;

    if (status.error) {
      box.innerHTML = `<span class="error">${status.error}</span>`;
      return;
    }
    if (status.loading) {
      // While loading, the message is the useful part; fps counters are all zero.
      box.innerHTML = `${status.message}&hellip;`;
      return;
    }
    const server = status.server || {};
    box.innerHTML =
      `<strong>${(status.detectFps || 0).toFixed(0)}</strong> fps &middot; ` +
      `sent <strong>${status.sent}</strong> &middot; ` +
      `skipped <strong>${status.clientDropped + (server.dropped || 0)}</strong> &middot; ` +
      `${status.delegate}${status.vendored ? ' &middot; local assets' : ''}`;
  };
}

/** Keep a canvas sized to its container in device pixels. */
export function fitCanvas(canvas) {
  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
  };
  resize();
  window.addEventListener('resize', resize);
  return resize;
}
