# 4. Rep counter — counting a repeated motion from pose

Builds exercise-style counting from MediaPipe Pose. The most instructive demo in the
set, because the obvious algorithm is the wrong one and you can feel the difference.

Reference implementation: [`demos/six_seven_counter/`](../../demos/six_seven_counter/)

## Prompt

```
Build a rep counter for an alternating-hands motion: as one hand rises the other
drops, like a see-saw, over and over. Count each swap.
Python 3.12, opencv-python + mediapipe, webcam at index 0.

Behaviour:
- MediaPipe Pose tracks my upper body and the count is drawn large on the frame.
- Draw what the counter is actually measuring, so I can see why it counts or does not.
- A "prepare" phase first: do not start counting until the camera can see both
  shoulders and both hands, held steady for about a second, and tell me what is
  missing if it cannot ("show both hands", "step back").
- Keys: r reset, q quit.

Rules:
- Do not compare each wrist against a fixed height. Instead track the *difference*
  between the two wrist heights and count each time its sign flips. That way there is
  nothing to calibrate, moving up or down in frame cancels out, and pumping one hand
  twice cannot score.
- Normalise that difference by a body measurement so my distance from the camera does
  not matter.
- Use a deadband: hands roughly level should hold the previous state rather than
  flicker between the two.
- Losing tracking for a frame or two must not reset the count, and must not force the
  prepare phase again.
- Pure logic class, tunables in a dataclass, and tests using synthetic pose landmarks.
```

## Follow-ups worth asking

* "It counted two reps when I did one." Usually a missing deadband, or counting both
  the rise and the fall.
* "It stops counting when I lean back." A threshold in pixels, or measured against the
  frame instead of the body.
* "It lost my reps when I stepped out of shot." Never reset a score on tracking loss;
  freeze instead.
* "One dropped frame sent it back to the prepare screen." Add a grace period before
  demanding a new hold.
* "Requiring my hips in frame makes me stand too far away." Ask for the scale
  reference to fall back to shoulder width when the hips are not visible.

## The lesson: pick a signal that cannot drift

The first version of this demo compared each wrist to a line above the shoulders. That
needed a threshold tuned per person, broke when the user sat down, and needed a
separate rule to stop one hand scoring twice.

Switching to *the sign of the difference between the wrists* deleted all three
problems at once. No calibration, immune to whole-body movement, and alternation
became structural — a sign flip is impossible unless both hands take part.

When a counter feels fragile, the fix is usually a better signal, not a better
threshold.
