# 1. Image lab — an interactive OpenCV playground

Builds a single-window app where you load an image, stack OpenCV operations, and see
the exact `cv2` code for whatever you just did. The best first project: you learn the
API by moving sliders instead of reading docs.

Reference implementation: [`demos/image_lab/`](../../demos/image_lab/)

## Prompt

```
I'm learning OpenCV and want an interactive playground to explore what each function
actually does. Python 3.12 in a venv, no GPU, opencv-python already installed.

Build a single-file script that:

- Loads an image from a path argument, or falls back to a generated test image so it
  runs with no arguments.
- Shows the original and the processed result side by side in one OpenCV window.
- Lets me pick an operation with the n / p keys and adjust every parameter of that
  operation with cv2 trackbars.
- Prints the equivalent cv2 code, with my current parameter values filled in, when I
  press r.

Cover at least: grayscale, Gaussian blur, median blur, fixed and adaptive threshold,
Canny, Sobel, morphology (erode/dilate/open/close), resize, rotate, and findContours
with bounding boxes and centroids.

Rules:
- Define the operations as data, not as a long if/elif chain. Each entry should know
  its name, its parameters with valid ranges, how to run itself, and how to describe
  itself as code. Adding an operation should mean adding one entry.
- The code the r key prints must be generated from the same parameter values used to
  render the preview, so it can never disagree with what I am looking at.
- Kernel size parameters must be forced odd, since OpenCV rejects even kernels.
- Any operation that needs a single channel should convert first and show that
  conversion in the printed code, so the snippet runs standalone.

Tell me which operations you could not test and why.
```

## What you should get back

A script with an operation table, a trackbar window, and a `r` key that prints
something like:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 80, 180, apertureSize=3)
img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
```

## Follow-ups worth asking

* "Let me chain several operations instead of one at a time, applied in order."
* "Add drawing operations: rectangle, circle, line, polygon with vertex markers, and
  text with a `getTextSize` backdrop." — drawing is where `cv2` gets genuinely useful
  for annotating detections later.
* "Store drawing coordinates as fractions of the image size so a shape stays put if I
  resize earlier in the chain, and show that arithmetic in the generated code."
* "A slider combination just crashed it with a cv2 error. Skip a failing step and
  keep the preview instead of losing the image."
* "Generate sample images that make each operation interesting: shapes for contours,
  a noisy gradient for the blurs, an unevenly lit document that defeats a global
  threshold but not an adaptive one."

## Why the "operations as data" rule matters

It is the difference between a demo and a teaching tool. Once each operation carries
its own parameter schema and its own code template, the UI, the code output and the
tests all derive from one table — so they cannot drift apart, and a participant adding
`cv2.bilateralFilter` touches exactly one place.
