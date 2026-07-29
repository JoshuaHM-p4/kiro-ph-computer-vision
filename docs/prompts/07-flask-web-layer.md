# 7. Flask web layer — put your demo in a browser

Takes a working desktop demo and gives it a web front end. There are three ways to do
this and picking the wrong one costs you an evening, so this prompt starts with the
choice.

Reference implementations: [`demos/common/webapp.py`](../../demos/common/webapp.py)
(the shared adapter), plus each demo's `web.py` and `templates/`.

## Pick your architecture first

| Approach | Vision runs | Good when | Cost |
|---|---|---|---|
| **A. Server-side MJPEG** | On the server | Simplest possible; one user | Server owns the webcam, single client only |
| **B. Vision in the browser** | In the browser (MediaPipe JS), landmarks streamed to the server | MediaPipe demos, multiple users, remote access | Two code paths to keep consistent |
| **C. Frames to the server** | On the server, browser posts JPEGs | The model has no browser build (YOLO, custom torch) | Bandwidth and latency per frame |

The MediaPipe demos here use **B**; the image lab and scavenger hunt use **C**. Choose
before you prompt, and say which one you picked.

## Prompt (approach B: vision in the browser)

```
I have a working desktop demo whose logic is a pure class: it takes normalized
landmarks plus a timestamp and returns a state dict. Add a Flask front end for it.

Architecture: MediaPipe runs in the *browser* via @mediapipe/tasks-vision. The browser
streams landmarks to Flask over a WebSocket as JSON; Flask owns the gesture logic and
answers with state; the browser draws the overlay from that state.

Requirements:
- Use flask-sock, not flask-socketio: it is WSGI-native so the plain Flask dev server
  serves WebSockets with no eventlet or gevent.
- Reuse the existing logic class untouched. The WebSocket handler should be a thin
  adapter, exactly like the desktop loop is.
- Each browser session gets its own state, keyed by an id the client generates, so two
  tabs do not fight.
- Every message carries an increasing sequence number. Drop anything not newer and
  reply with the previous state marked as skipped, so a congested socket degrades into
  a lower frame rate instead of corrupted state.
- The browser should skip sending a frame while the socket still has buffered bytes,
  rather than queueing stale landmarks.
- Add an on-demand /snapshot route that renders the authoritative result server-side
  with cv2 and returns a PNG, so what I download matches what I saw.
- Bind to 127.0.0.1 and warn me if I ever pass 0.0.0.0 — this drives a webcam and has
  no authentication.
- Show the connection state and frame rate in the page so I can tell whether it is
  working.

Also give me an HTTP POST fallback for the same payload, for environments where
WebSockets are blocked.
```

## Prompt (approach C: frames to the server)

```
Add a Flask front end for a demo whose model only runs in Python (YOLO).

The browser captures webcam frames, posts them as JPEG about 4 times a second, and the
server runs inference and returns detections plus game state as JSON. No WebSocket
needed: one frame is one POST whose response carries everything.

Requirements:
- Throttle in the browser: never have two requests in flight, and skip a frame if the
  previous one has not returned.
- JPEG quality around 0.6 — this is inference input, not a photograph.
- Also accept a full-resolution upload from a file picker, for a single still.
- An undecodable frame must return a normal response with an error field, never a 500,
  and must not stall the game clock.
- The server owns the clock. Do not trust timestamps from the client.
- Per-session state keyed by a client-generated id.
```

## Follow-ups worth asking

* "The page sits on 'loading models' forever." Almost always a cold-cache CDN download
  of 10-15 MB. Ask for per-stage progress messages, a warning after a few seconds, and
  a hard timeout that names the stage that stalled. Then ask for a tool that vendors
  the assets locally — it turns a 50 second first load into 2 seconds and works offline.
* "It works on my machine but not from my phone." Browsers only grant camera access on
  `localhost` or HTTPS. That is a browser rule, not a bug in your code.
* "The overlay lags behind the video." Draw from the landmarks you just sent rather
  than waiting for the round trip.
* "Add a settings panel." Then make sure the settings are per-session and clamped on
  the server, or one visitor changes everyone's experience.

## The thing worth copying

Because the logic class is pure, the desktop app and the web app are both thin
adapters over the *same* implementation. Behaviour cannot drift between them, and the
whole test suite runs against the logic without a browser or a camera. Ask for that
shape explicitly:

> The Flask route and the desktop loop must both be thin adapters over the same logic
> class. Do not reimplement any decision-making in the web layer.
