# 2. Air canvas — paint with your fingertip

Builds hand tracking plus a drawing surface plus a gesture-driven palette. The classic
MediaPipe demo, and a good introduction to turning landmarks into intent.

Reference implementation: [`demos/air_canvas/`](../../demos/air_canvas/)

## Prompt

```
Build an "air canvas": I paint on the webcam feed with my fingertip.
Python 3.12, no GPU, opencv-python and mediapipe installed. Webcam at index 0.

Behaviour:
- MediaPipe Hands tracks one hand and I draw with my index fingertip.
- Index finger extended alone = draw. Index + middle extended = move a cursor without
  drawing. Thumb-to-index pinch = erase. Fist = do nothing.
- A palette down the left side: several colours, an eraser, undo and clear. I select a
  cell by resting the cursor on it, not by touching it, so sweeping past does nothing.
- Two vertical sliders on the right for brush size and opacity.
- Keys: c clear, z undo, s save the painting as a PNG, q quit.

Rules:
- Put the paint logic in a pure class that takes normalized landmarks plus a timestamp
  and returns state. No camera, no cv2 window, no drawing calls inside it.
- Landmarks stay in 0..1 everywhere. Convert to pixels only when drawing.
- Gesture thresholds must be relative to hand size, not pixels, so my distance from
  the camera does not change how hard it is to pinch.
- Brush size is a fraction of frame height, so a stroke looks the same at any
  resolution.
- Opacity must be real alpha compositing, not a lighter colour.
- Every threshold in a dataclass at the top of the file.

Then tell me what you could not verify without a camera.
```

## Follow-ups worth asking

* "The stroke has hard corners and jitters when my hand is still. Smooth the fingertip
  with an exponential moving average, and skip points closer together than a minimum
  distance."
* "Pinching to erase also triggers when I make a fist. Only treat it as a pinch if the
  index finger is actually extended." — a real bug: a closed fist brings the thumb and
  index tips together too.
* "A stroke at 40% opacity gets darker where it crosses itself. Composite each stroke
  on its own layer so overlapping segments within one stroke do not stack alpha."
* "Show the gesture map on screen and highlight whichever gesture is currently
  recognised, so it doubles as feedback that the gesture landed."
* "Put the slider labels and their current values *under* the bars, so my fingertip
  does not cover the number I am trying to read."

## The two mistakes everyone makes here

**Pixel thresholds.** "Pinch when the tips are within 40 pixels" works at the distance
you tested and nowhere else. Divide by the hand span and the problem disappears.

**Instant selection.** A palette that activates on entry fires for every cell your
finger crosses on the way. Requiring the cursor to dwell for ~0.4 s fixes it, and the
dwell progress ring is nice feedback.
