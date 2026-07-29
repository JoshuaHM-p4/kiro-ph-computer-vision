# Prompt library

Every demo in [`demos/`](../../demos/) was built by prompting Kiro. These are the
prompts, cleaned up so you can paste them into your own session, change the subject
matter, and get something comparable for *your* idea.

Use them as starting points, not scripture. The interesting part is the shape: what
context to give, which constraints to state up front, and what to ask for in return.

## Start here

| # | Prompt | Builds | Teaches |
|---|---|---|---|
| — | [How to prompt for vision code](00-how-to-prompt.md) | — | The pattern all the others follow |
| 1 | [Image lab](01-image-lab.md) | An interactive `cv2` playground with a code-reveal button | The OpenCV API itself: filters, thresholds, contours, drawing |
| 2 | [Air canvas](02-air-canvas.md) | Painting with a fingertip and a gesture palette | Hand landmarks, gesture states, drawing surfaces |
| 3 | [Slide presenter](03-slide-presenter.md) | A laser pointer and pinch-to-advance controller | Handedness, edge-triggered actions, cooldowns |
| 4 | [Rep counter](04-rep-counter.md) | Counting alternating-hand reps from body pose | Pose landmarks, and choosing a signal that cannot drift |
| 5 | [PNGTuber](05-pngtuber.md) | An avatar driven by head angle and expression | Head pose, facial ratios, calibration |
| 6 | [Scavenger hunt](06-scavenger-hunt.md) | A timed game over YOLO object detection | Pretrained detection models, game state, scoring |
| 7 | [Flask web layer](07-flask-web-layer.md) | Any of the above, in a browser | Streaming vision data to a server, or frames to a model |
| 8 | [Testing vision code](08-testing-vision-code.md) | A test suite that needs no webcam | Making camera code testable at all |

## How to use one

1. Pick the prompt closest to your idea and copy the **Prompt** block.
2. Replace the subject matter with yours. Keep the *constraints* — they are what stop
   the output turning into an unmaintainable single file.
3. Paste it into Kiro from inside your project folder:
   ```bash
   cd projects/<your-username>
   kiro-cli chat
   ```
4. Read what comes back and push on it. The **Follow-ups worth asking** section in
   each file lists the questions that actually improved these demos.

## A warning about copy-pasting

These prompts produce a *starting point*, not a finished demo. Every demo in this
repository needed several rounds of "that flickers", "that only works when I stand
close", "that counted twice". The follow-up rounds are where the real work happened,
and each prompt file records the ones that mattered.
