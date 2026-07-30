# Computer vision patterns

Conventions for writing vision code in this project that Kiro should follow.

## Separate I/O from logic

The camera loop and the display code must not contain decision-making. Logic goes
in functions/classes that take numbers (landmarks, timestamps) and return state.

This is the single highest-leverage rule: it makes the code testable without a
webcam, and it is what lets the desktop window and the Flask web app share one
implementation.

## Normalized coordinates

Keep landmarks in 0..1 and convert to pixels only at draw time with
`geometry.to_pixels(point, width, height)`. This prevents resolution bugs.

## Scale-relative thresholds

Express thresholds as a fraction of:
- **hand span** (wrist to middle knuckle) for hand gestures
- **interocular distance** for face ratios
- **torso length** (shoulder to hip, or shoulder width fallback) for body pose

Never use pixel thresholds. Moving closer to the camera must not change behaviour.

## Hysteresis on anything that toggles

Use separate enter and release thresholds (see `geometry.HysteresisLatch`). A value
hovering at a single threshold flickers the state every frame.

## Dwell to activate menus

Palette cells and menu items activate after a sustained hover (see
`geometry.DwellTimer`), not on entry. Otherwise a finger sweeping across a rail
fires for every cell it passes.

## Tunables in one place

Every threshold goes in the demo's `config.py` dataclass, not inline in the logic.
This makes them findable and adjustable from CLI flags.

## Testing without a camera

Build synthetic fixtures with `demos/tests/fixtures.py`:
- `make_hand(pinch=0.05, extended=("index",))`
- `make_face(yaw=30, mouth_open=0.4)`
- `make_pose(left_wrist_up=0.3)`

Drive the pure logic class with those and a fake clock. Never sleep in a test.
Guard fixture values against threshold changes with a dedicated assertion.

## When helping a participant

1. Check `docs/prompts/` first — the answer is often already there.
2. Start with the simplest thing that works: one file, one operation.
3. Split I/O from logic once it works.
4. Add tests for the logic once it is split.
5. Only then suggest a web layer.
