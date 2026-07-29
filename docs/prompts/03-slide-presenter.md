# 3. Slide presenter — laser pointer and pinch to advance

Builds gesture control of something real. Short, satisfying, and it teaches the two
things that bite everyone: handedness and repeated triggers.

Reference implementation: [`demos/slide_presenter/`](../../demos/slide_presenter/)

## Prompt

```
Build a gesture-controlled slide presenter. Python 3.12, opencv-python + mediapipe,
webcam at index 0.

Behaviour:
- Loads slides from a folder of PNG/JPG images, in natural filename order so slide2
  comes before slide10.
- The current slide fills the window, with a small webcam picture-in-picture in the
  bottom-left corner so I can see what the tracker sees.
- Pointing with my index finger puts a glowing laser dot on the slide.
- Pinching with my right hand advances one slide; pinching with my left goes back.
- Keys as a fallback: arrow keys or n/p, r to reload the folder, q to quit.

Rules:
- Slide state and laser position go in a pure class: landmarks and a timestamp in,
  state out. No cv2 calls inside it.
- One pinch must move exactly one slide, no matter how long I hold it. Use an
  edge-triggered action plus a cooldown.
- Track both hands independently, so holding a pinch on one hand cannot block the
  other.
- Clamp at the deck ends, with an option to wrap around.
- Hold the laser dot briefly when tracking drops a frame, so it does not blink.

Tell me what you could not verify without a camera.
```

## Follow-ups worth asking

* "Next and previous are swapped." — See below. Expect this one.
* "One pinch advanced three slides." Means the action is level-triggered rather than
  edge-triggered: it fires every frame the pinch is held.
* "The laser is jittery." Smooth the fingertip, and keep the smoothing separate from
  the pinch detection so lag in one does not affect the other.
* "Make a `w` key that swaps the hand mapping live, so I can fix it without editing
  code."
* "Generate a placeholder deck so the demo runs before I have slides."

## Handedness will be backwards, and here is why

MediaPipe assigns "Left" and "Right" assuming the image is a **selfie mirror**. So the
label depends on whether you mirrored the frame *before* detection:

* Mirror the frame, then detect: the label matches the hand you see on screen.
* Detect on the raw feed: the label is the opposite of what you expect.

In a browser this bites twice, because `getUserMedia` gives you an unmirrored feed
while the CSS transform makes it *look* mirrored. Do not try to reason your way to the
right answer — put a runtime swap flag in the config, try it, and flip it if it feels
wrong. Every one of these demos ships that flag for exactly this reason.
