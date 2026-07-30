# 9. SAM labeler — segment anything you can name

Builds a text-prompted segmentation tool: capture or upload a picture, type "coffee
mug", get a mask. Then it becomes an OpenCV exercise — what do you *do* with a mask?

Teaches gated models, credential handling, and the fact that a segmentation model is
only half of a useful tool.

Reference implementation: [`demos/sam_labeler/`](../../demos/sam_labeler/)

## Prompt

```
Build a text-prompted image labeler using SAM 3.1 (facebook/sam3) via transformers.
Python 3.12, CPU only.

Behaviour:
- I get a picture two ways: capture a still from the webcam, or upload a file.
- I type what to look for in plain words, one or several: "person, laptop, mug".
- SAM returns a mask per match. Show each one over the image, and list the labels it
  found with a count each.
- Per label I can pick a colour and an effect: fill, outline, blur inside the mask,
  pixelate, spotlight (darken everything else), cutout, or hide.
- A mode switch between segmentation (draw the mask) and detection (reduce each mask
  to its bounding box).
- Save the annotated result as a PNG.

Rules:
- Keep the rendering pure: a function that takes a BGR image, a list of
  (label, score, mask), and a style per label, and returns a new image. No model and
  no camera inside it, so I can test every effect with synthetic masks.
- facebook/sam3 is gated. Take a Hugging Face read token, hold it in memory only,
  never write it to disk or a log, and never include it in any response. Prefer
  HF_TOKEN from the environment if it is set.
- If transformers is too old to have Sam3Model, or there is no token, or the download
  fails: say so clearly and keep working. Do not crash on import.
- Add a demo mode that produces synthetic masks so the effects and the UI can be
  explored with no model and no token at all.
- Downscale big uploads before inference; SAM on a CPU is slow.
- Effects that act on everything *outside* a mask (spotlight, cutout) must be applied
  once over the union of those masks, not per instance.

Tell me the download size before installing anything, and tell me what you could not
verify.
```

## Follow-ups worth asking

* "Draw the labels largest-mask-first so a small object inside a big one stays visible."
* "Label chips are unreadable on light colours." Compute the luminance of the chip and
  switch the text between black and white.
* "Which of these settings need a re-run and which just redraw?" Score and mask
  thresholds change *inference*; opacity, thickness, blur and pixel size only change
  *drawing*. Say so in the UI or people will wait for nothing.
* "Export the masks as YOLO polygons so I can train on them." That is the natural next
  step and the reason this pattern exists — SAM labels, you train something small.

## Credentials are part of the design, not an afterthought

A gated model means your demo now handles a secret. State the rules in the prompt:

> Hold the token in memory only, never write it to disk or a log, never return it in a
> response, and mask the input field.

Worth knowing: it is easy to leak one by accident. `echo ${HF_TOKEN:-no}` prints the
token, not the word "no". Prefer `${HF_TOKEN:+set}` when you just want to check whether
a variable exists, and if a token ever lands in a terminal transcript, rotate it.

## The bigger lesson: a model is half a tool

SAM gives you a boolean array. Everything that makes it *useful* is OpenCV:

```python
# blur a face
frame[mask] = cv2.GaussianBlur(frame, (31, 31), 0)[mask]

# spotlight a subject
frame[~mask] = (frame[~mask] * 0.3).astype(np.uint8)

# outline for a diagram
contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(frame, contours, -1, colour, 3)
```

Three lines each, and they are the difference between a research demo and something
you would actually ship. If you only take one thing from this repository, take the
habit of asking "and then what do I do with the output?"
