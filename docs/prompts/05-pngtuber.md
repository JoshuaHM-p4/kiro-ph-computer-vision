# 5. PNGTuber — an avatar driven by your face

Builds head-pose estimation and expression classification from Face Mesh, driving a
sprite. Teaches you to derive meaning from landmark *ratios*, and why a per-user
baseline beats absolute numbers.

Reference implementation: [`demos/pngtuber/`](../../demos/pngtuber/)

## Prompt

```
Build a PNGTuber: an avatar image that changes with my head angle and facial
expression. Python 3.12, opencv-python + mediapipe, webcam at index 0.

Behaviour:
- MediaPipe Face Mesh tracks my face.
- Head yaw picks one of three sprites: turned left, facing forward, turned right.
- Facial expression picks between neutral, happy, surprised and angry.
- So twelve sprites in total, named {yaw}_{expression}.png, alpha-composited over the
  webcam or over a solid background I can toggle.
- Generate placeholder sprites programmatically so it runs before I have artwork.
- Keys: c recalibrate, b cycle background, 1-4 preview an expression, q quit.

Rules:
- Estimate yaw in degrees with cv2.solvePnP against a canonical 3D face model, and
  fall back to a simpler nose-offset ratio if the solve fails.
- Infer expressions from landmark ratios: mouth aspect ratio, mouth-corner lift,
  brow-to-eye distance, eye aspect ratio. Divide each by a face measurement so they do
  not change with my distance from the camera.
- Compare those ratios against a *neutral baseline captured from me* over the first
  second, not against hardcoded numbers. My resting face is not the same as yours.
- Use hysteresis on the yaw buckets and require an expression to persist briefly
  before the sprite changes, so nothing flickers.
- Pure classifier, tunables in a dataclass, tests with synthetic face landmarks.
```

## Follow-ups worth asking

* "The sprite flickers between forward and left when I hold still." Yaw needs separate
  enter and release angles.
* "Everything reads as one expression." The baseline captured a non-neutral face — add
  a recalibrate key and say so on screen while calibrating.
* "It says I look surprised whenever I turn my head." See below; this is the good bug.
* "Add a chroma-green background mode so I can key it into streaming software."

## The good bug: yaw contaminating expression

Every expression ratio here divides a *vertical* distance by a *horizontal* one — brow
height over eye width, lip gap over mouth width. Turning your head foreshortens the
horizontal denominator by `cos(yaw)`, so every ratio inflates. At about 40 degrees a
completely relaxed face crosses the "surprised" threshold.

The fix is one line — multiply the ratios back by `cos(yaw)` — but you have to notice
it first, and you only notice it by testing the two variables *together*. Worth asking
for explicitly:

> Check whether turning my head changes the detected expression, and if it does,
> compensate for the foreshortening.

There is a second-order trap inside it: use the *current frame's* yaw for that
compensation, not a smoothed value. During a fast turn the smoothed yaw lags, so the
correction under-shoots and the avatar flashes the wrong face.
