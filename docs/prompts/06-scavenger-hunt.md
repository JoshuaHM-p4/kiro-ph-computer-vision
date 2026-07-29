# 6. Scavenger hunt — a game over object detection

Builds a timed game on top of a pretrained YOLO model: "bring me a cup" appears, you
hold one up before the clock runs out. Proves a pretrained model works on your own
desk, with no dataset of your own.

Reference implementation: [`demos/scavenger_hunt/`](../../demos/scavenger_hunt/)

## Prompt

```
Build a scavenger hunt game using a pretrained object detection model.
Python 3.12, no GPU, opencv-python installed. Webcam at index 0.

Behaviour:
- Use Ultralytics YOLO with a COCO-pretrained model. Let it download the weights on
  first use so I do not have to find a model file.
- "BRING ME A CELL PHONE" appears over the webcam with a 30 second countdown. I hold
  the real object up to the camera; when the model sees it, I score.
- Targets are drawn from the COCO classes I plausibly have at a desk: cup, cell phone,
  scissors, book, keyboard, remote, potted plant, banana. Not "giraffe".
- Draw every detection as a labelled box, with the target highlighted.
- Five rounds, points for speed, a streak bonus, and a scoreboard at the end.
- Keys: SPACE start, n skip the item, q quit.

Rules:
- Game logic in a pure class: a list of detections plus a timestamp in, state out. No
  model and no camera inside it, so I can test the rules with fake detections.
- The target must be visible for a short hold before it counts, so one flukey frame
  cannot win a round.
- Never crash if the model cannot load. Fall back to a mode that explains what went
  wrong and still runs.
- Weights must not end up committed. Put them somewhere git-ignored.
- Run detection every Nth frame and reuse the last result in between, since a CPU
  cannot keep up at 30 fps.

Warn me before installing anything large, and tell me the download size.
```

## Follow-ups worth asking

* "It scored the moment something flickered in frame." Lengthen the hold window.
* "Let me upload a photo instead of holding the object up." Note that a still cannot be
  "held", so it needs a separate path that scores on one confident detection.
* "Add a settings panel for the timer and the confidence threshold." Then: "changing
  the timer mid-round instantly ended my round" — settings that affect timing should
  apply from the *next* round.
* "How do I know the model is actually working?" Ask it to run detection on a photo
  with known contents and print the labels and confidences.

## The install trap, and how to avoid it

`pip install ultralytics` pulls in torch, and plain PyPI gives you the **CUDA** build:
3-5 GB of nvidia wheels you do not need for a webcam game on a laptop. The CPU build
is about 250 MB and identical for this purpose:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install ultralytics
```

Ask for this explicitly — "prefer CPU wheels and tell me the download size before
installing" — or you will wait a long time for gigabytes you will never execute.

## If you want to avoid torch entirely

Export the model to ONNX and run it through OpenCV's own DNN module, which you already
have:

> Add a second detection backend that loads a YOLO .onnx export with cv2.dnn, so the
> demo can run without torch. Sniff the output tensor shape rather than assuming a
> layout, because it changed between YOLO generations.
