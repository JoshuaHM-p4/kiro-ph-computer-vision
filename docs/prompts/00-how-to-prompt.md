# How to prompt for computer vision code

The pattern every prompt in this folder follows, and why each part earns its place.

## The shape

```
1. Context   what you have, what you are running on
2. Goal      one sentence on the behaviour you want
3. Inputs    where the pixels come from
4. Rules     the constraints you refuse to compromise on
5. Output    the files or functions you expect back
6. Done      how you will know it works
```

You can drop parts 4-6 and still get code. You will just get code you then have to
argue with.

## Why the constraints matter more than the goal

"Make a hand-tracking paint app" gets you a 300-line file with the camera loop, the
gesture thresholds, the drawing and the UI all interleaved. It demos fine and then
resists every change you want to make.

Adding one rule changes the whole result:

> Keep the gesture logic in a pure function that takes landmarks and returns state.
> No camera, no window, no drawing inside it.

Now the thing you want to tune is separable from the thing that needs a webcam, and
you can test it. Every demo here is built that way, and it is the single highest
leverage sentence in any of these prompts.

## Constraints worth reusing verbatim

* **Separate I/O from logic.** "The camera loop and the display code must not contain
  any decision-making. Logic goes in functions that take numbers and return numbers."
* **Normalized coordinates.** "Keep landmarks in 0..1 and convert to pixels only at
  draw time." Saves you a resolution bug later.
* **Scale-relative thresholds.** "Express thresholds as a fraction of hand span / eye
  distance / torso length, never in pixels, so moving closer to the camera does not
  change behaviour." This one fixes a bug you would otherwise spend an evening on.
* **Hysteresis on anything that toggles.** "Use separate enter and release thresholds
  so a value hovering at the boundary cannot flicker."
* **Tunables in one place.** "Put every threshold in a dataclass at the top, not
  inline in the logic."
* **Tell me what you could not verify.** Keeps the summary honest about which parts
  actually ran.

## Say what hardware you are on

Vision libraries are unusually sensitive to versions and platform. Put this in the
first message:

> Python 3.12 in a venv, on a laptop with no GPU. Webcam at /dev/video0. Prefer CPU
> inference. Pin dependency versions and tell me if a package needs a specific
> Python version.

That single paragraph prevents the two most common workshop failures: a package with
no wheel for your interpreter, and a multi-gigabyte CUDA download you did not want.

## Iterate on behaviour, not code

The useful follow-ups describe what you *see*, not what you think is wrong in the
code:

* "The brush jumps around when my hand is still."
* "It counts two reps when I only did one."
* "It works when I sit close but not when I lean back."
* "The sprite flickers between two poses when I hold my head still."

Each of those has a standard fix (smoothing, edge-triggering, scale-relative
thresholds, hysteresis) and Kiro will reach for it if you describe the symptom
plainly. Describing the symptom also gives you a test case to keep.

## Ask for tests you can run without a webcam

> Write tests that feed synthetic landmark arrays instead of using a camera, so the
> suite runs in CI.

This is what makes vision code maintainable. See
[08-testing-vision-code.md](08-testing-vision-code.md).
