# projects/

**This is where you build.** Make a folder named after your GitHub username and put
your code in it:

```bash
mkdir projects/<your-github-username>
cd projects/<your-github-username>
kiro-cli chat
```

Everything else in this repository is reference material. Nothing here is graded on
structure, so organise your folder however suits your project — but if you would like a
starting point, [`your_example_app/`](your_example_app/) shows a shape that works.

## Why a folder per person

Everyone forks the same repository, so a folder per participant means:

* You can pull updates to the shared parts without conflicts.
* Your project is a self-contained thing you can lift out afterwards.
* Comparing approaches at showcase time is just opening two folders.

## What to put in it

At minimum a script you can run and a line saying how to run it. A `README.md` in your
folder saying what you built and what you learned is worth more than tidy code — this
is a build night, not a code review.

## Where to look when you are stuck

| Question | Look at |
|---|---|
| How do I ask Kiro for this? | [`docs/prompts/`](../docs/prompts/) |
| How did the demos do it? | [`demos/`](../demos/) and [`demos/README.md`](../demos/README.md) |
| What OpenCV function do I want? | Run the image lab: `python -m demos.image_lab.desktop --list` |
| My gesture flickers / drifts / double-counts | [`docs/prompts/00-how-to-prompt.md`](../docs/prompts/00-how-to-prompt.md) |
