# 8. Testing vision code without a webcam

The prompt that makes everything else maintainable. A vision project you cannot test is
a project you can only change by trying it and squinting.

Reference implementation: [`demos/tests/`](../../demos/tests/) — 674 tests, no camera,
no model, no internet.

## Prompt

```
Write a pytest suite for my vision demo that runs with no webcam, no model download
and no internet, so it works in CI.

Approach:
- Build synthetic landmark fixtures: functions that produce anatomically plausible
  hand / face / pose landmark arrays with parameters I can control, e.g.
  make_hand(pinch=0.05, extended=("index",)), make_face(yaw=30, mouth_open=0.4),
  make_pose(left_wrist_up=0.3).
- Drive the pure logic class with those fixtures and a fake clock, asserting on the
  state it returns. Never sleep in a test.
- For the rendering code, assert on properties rather than pixels: output shape and
  dtype unchanged, the frame actually modified, no exception when landmarks fall
  outside the frame.
- For Flask routes, use the test client. Assert the JSON contract, session isolation
  between two client ids, and that malformed input returns a handled error rather than
  a 500.
- Cover the failure modes explicitly: no hand in frame, tracking lost mid-gesture, a
  value hovering exactly at a threshold, an object at the very edge of the frame.

Write the tests so a future threshold change fails loudly rather than silently
disabling them.
```

## The tests that earn their keep

Not "does it detect a hand" — that needs a camera. These:

* **Hysteresis holds.** Feed 60 frames of a value oscillating just inside the deadband
  and assert the count stays zero.
* **One gesture, one action.** Hold a pinch for 20 frames, assert the slide advanced
  exactly once.
* **Scale invariance.** Run the same gesture with a hand span of 0.3 and 0.06 and
  assert identical results. This is the test that catches pixel thresholds.
* **Tracking loss.** Drop the landmarks mid-gesture and assert the score survives and
  the state does not latch.
* **The bug you just fixed.** Every "it counted twice" becomes a permanent test.

## The trap: tests that pass for the wrong reason

Twice while building these demos a threshold change quietly broke the fixtures — the
synthetic values no longer straddled the new thresholds, so rep tests passed with a
count of zero, and a wrist pushed outside the frame passed a visibility check by
failing it. Both looked green.

Guard against it:

> Add a test asserting that my synthetic fixture values sit outside the configured
> thresholds, so changing a threshold breaks that test instead of silently disabling
> the others.

## Run it

```bash
.venv/bin/python -m pytest projects/<your-username>
```
