"""Scavenger Hunt — camera loop and UI.

All the messy I/O lives here; the rules live in logic.py.

Run from the projects/scavenger_hunt directory (or the repo root):

    python projects/scavenger_hunt/main.py
    python projects/scavenger_hunt/main.py --camera 1
    python projects/scavenger_hunt/main.py --detect-every 5

Controls
--------
  SPACE    start the game
  n        skip the current item
  s        open / close the settings panel
  c        cycle to the next available camera
  +/-      raise / lower the round timer by 5 s  (takes effect next round)
  [/]      lower / raise the confidence threshold by 5 %  (takes effect immediately)
  q / ESC  quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Resolve import path so the script runs both from the project dir and from
# the repo root (python projects/scavenger_hunt/main.py).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from logic import (  # noqa: E402
    DONE, FOUND, PLAYING, SKIP, WAITING,
    Config, ScavengerHunt,
)

# ---------------------------------------------------------------------------
# Colours  (BGR)
# ---------------------------------------------------------------------------
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,   0)
C_GREEN   = (50,  220, 50)
C_RED     = (50,  50,  220)
C_YELLOW  = (0,   220, 220)
C_CYAN    = (220, 220, 0)
C_ORANGE  = (0,   160, 255)
C_GREY    = (160, 160, 160)
C_DARK    = (20,  20,  20)
C_PURPLE  = (200, 80,  200)
C_TARGET  = (0,   255, 0)    # bright green box on target object
C_OTHER   = (180, 180, 60)   # dim box on everything else

# Accent gradient stops (BGR) for the top bar
C_GRAD_A  = (180, 60,  20)   # deep blue-ish
C_GRAD_B  = (40,  160, 220)  # warm amber

FONT      = cv2.FONT_HERSHEY_SIMPLEX
WIN_NAME  = "Scavenger Hunt"


# ---------------------------------------------------------------------------
# YOLO loader — graceful fallback if ultralytics is not installed
# ---------------------------------------------------------------------------

class _DummyModel:
    """Returned when ultralytics cannot be imported or the model fails to load."""

    def __init__(self, reason: str):
        self.reason = reason
        self.names: dict[int, str] = {}

    def __call__(self, frame: np.ndarray, verbose: bool = False):
        return []


def _load_model(weights_dir: Path) -> tuple[object, str | None]:
    """
    Load YOLOv8n, saving weights to weights_dir so they are git-ignored.
    Returns (model, error_message).  error_message is None on success.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        msg = (
            "ultralytics is not installed.\n"
            "Run:  pip install ultralytics\n"
            "(~200 MB download including torch CPU build)"
        )
        return _DummyModel(msg), msg

    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "yolov8n.pt"

    try:
        model = YOLO(str(weights_path) if weights_path.exists() else "yolov8n.pt")
        downloaded = Path("yolov8n.pt")
        if downloaded.exists() and not weights_path.exists():
            downloaded.rename(weights_path)
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not load YOLO model: {exc}"
        return _DummyModel(msg), msg

    return model, None


def _run_detection(model, frame: np.ndarray, confidence: float = 0.5) -> list[dict]:
    """
    Run inference and return a list of dicts:
      {"label": str, "conf": float, "box": (x1, y1, x2, y2)}
    Returns [] if model is a _DummyModel or inference fails.
    """
    if isinstance(model, _DummyModel):
        return []
    try:
        results = model(frame, verbose=False)
        out = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                label  = model.names[cls_id]
                conf   = float(boxes.conf[i].item())
                if conf < confidence:
                    continue
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                out.append({"label": label, "conf": conf, "box": (x1, y1, x2, y2)})
        return out
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Camera probe — find all usable camera indices at startup
# ---------------------------------------------------------------------------

def _probe_cameras(max_index: int = 4) -> list[int]:
    """
    Try to open VideoCapture for indices 0..max_index.
    Returns a list of indices that successfully opened.
    """
    available: list[int] = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available if available else [0]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _s(value: float, h: int, w: int, base_h: int = 480, base_w: int = 640) -> float:
    """Scale a size value relative to current frame dimensions."""
    scale = min(w / base_w, h / base_h)
    return max(value * scale, value * 0.5)


def _si(value: float, h: int, w: int) -> int:
    """Integer version of _s."""
    return max(1, int(_s(value, h, w)))


def _draw_gradient_rect(
    frame: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
    alpha: float = 0.72,
    horizontal: bool = True,
) -> None:
    """Fill a rectangle with a horizontal (or vertical) gradient, blended over the frame."""
    region = frame[y0:y1, x0:x1]
    h_r, w_r = region.shape[:2]
    if h_r <= 0 or w_r <= 0:
        return
    grad = np.zeros((h_r, w_r, 3), dtype=np.uint8)
    steps = w_r if horizontal else h_r
    for i in range(steps):
        t = i / max(steps - 1, 1)
        b = int(color_a[0] * (1 - t) + color_b[0] * t)
        g = int(color_a[1] * (1 - t) + color_b[1] * t)
        r = int(color_a[2] * (1 - t) + color_b[2] * t)
        if horizontal:
            grad[:, i] = (b, g, r)
        else:
            grad[i, :] = (b, g, r)
    blended = cv2.addWeighted(grad, alpha, region, 1 - alpha, 0)
    frame[y0:y1, x0:x1] = blended


def _draw_panel(
    frame: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
    bg: tuple[int, int, int] = C_DARK,
    border: tuple[int, int, int] = C_CYAN,
    alpha: float = 0.78,
    border_thickness: int = 1,
) -> None:
    """Draw a semi-transparent panel with a coloured border."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), border, border_thickness)


def _text_with_shadow(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw text with a dark shadow for legibility over any background."""
    x, y = pos
    cv2.putText(frame, text, (x + 2, y + 2), FONT, scale, C_BLACK,  thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x,     y),     FONT, scale, colour,   thickness,     cv2.LINE_AA)


def _draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    target: str,
) -> None:
    """Draw bounding boxes; highlight the target in bright green."""
    h, w = frame.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = d["label"]
        conf  = d["conf"]
        is_target = label == target

        colour    = C_TARGET if is_target else C_OTHER
        thickness = _si(3, h, w) if is_target else 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
        tag = f"{label} {conf:.0%}"
        _text_with_shadow(frame, tag, (x1, max(y1 - 6, 12)),
                          _s(0.5, h, w), colour)


def _draw_hud(
    frame: np.ndarray,
    state: dict,
    hold_progress: float,
    model_error: str | None,
    fps: float,
    cam_idx: int,
    available_cams: list[int],
    pulse: float,          # 0..1 animated value for border effects
) -> None:
    """Overlay all HUD elements onto frame in-place."""
    h, w = frame.shape[:2]
    phase = state["phase"]

    # ---- Animated top gradient bar ----
    bar_h = _si(52, h, w)
    _draw_gradient_rect(frame, 0, 0, w, bar_h, C_GRAD_A, C_GRAD_B, alpha=0.80)

    # Subtle pulsing border on the whole frame
    pulse_intensity = int(60 + 80 * pulse)
    pulse_color = (pulse_intensity, pulse_intensity, 255)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), pulse_color, 2)

    # ---- FPS counter (top-right corner) ----
    fps_txt = f"FPS {fps:.0f}"
    fps_scale = _s(0.45, h, w)
    (fw, fh), _ = cv2.getTextSize(fps_txt, FONT, fps_scale, 1)
    _text_with_shadow(frame, fps_txt, (w - fw - 10, bar_h - 6), fps_scale, C_GREY, 1)

    # ---- Camera label (below FPS) ----
    if len(available_cams) > 1:
        cam_txt = f"CAM {cam_idx}  [c=switch]"
    else:
        cam_txt = f"CAM {cam_idx}"
    cam_scale = _s(0.42, h, w)
    (cw, _), _ = cv2.getTextSize(cam_txt, FONT, cam_scale, 1)
    _text_with_shadow(frame, cam_txt, (w - cw - 10, bar_h + _si(16, h, w)),
                      cam_scale, C_PURPLE, 1)

    # ---- Model error banner ----
    if model_error:
        banner_lines = model_error.split("\n")
        y = bar_h + _si(22, h, w)
        for line in banner_lines:
            _text_with_shadow(frame, line, (10, y), _s(0.55, h, w), C_RED, 2)
            y += _si(24, h, w)

    # ---- WAITING splash ----
    if phase == WAITING:
        cx = w // 2
        cy = h // 2
        title_scale = _s(1.2, h, w)
        sub_scale   = _s(0.75, h, w)
        title = "SCAVENGER  HUNT"
        (tw, th), _ = cv2.getTextSize(title, FONT, title_scale, 3)
        _text_with_shadow(frame, title, (cx - tw // 2, cy - _si(30, h, w)),
                          title_scale, C_YELLOW, 3)
        sub = "Press  SPACE  to  start"
        (sw, _), _ = cv2.getTextSize(sub, FONT, sub_scale, 2)
        _text_with_shadow(frame, sub, (cx - sw // 2, cy + _si(30, h, w)),
                          sub_scale, C_WHITE, 2)
        return

    # ---- DONE scoreboard ----
    if phase == DONE:
        _draw_scoreboard(frame, state)
        return

    # ---- Round header inside top bar ----
    rnd  = state["round"]
    tot  = state["total_rounds"]
    tgt  = state["target"].upper()
    prompt = f"FIND:  {tgt}   ({rnd}/{tot})"
    prompt_scale = _s(0.80, h, w)
    (pw, _), _ = cv2.getTextSize(prompt, FONT, prompt_scale, 2)
    _text_with_shadow(frame, prompt, (10, bar_h - _si(10, h, w)),
                      prompt_scale, C_CYAN, 2)

    # ---- Score & streak row ----
    score_txt = f"Score: {state['score']}"
    streak = state["streak"]
    streak_txt = ""
    if streak > 1:
        streak_txt = f"  STREAK  x{streak}!"
    score_scale = _s(0.65, h, w)
    _text_with_shadow(frame, score_txt, (10, h - _si(54, h, w)),
                      score_scale, C_WHITE)
    if streak_txt:
        (sw2, _), _ = cv2.getTextSize(score_txt, FONT, score_scale, 2)
        _text_with_shadow(frame, streak_txt,
                          (10 + sw2 + _si(6, h, w), h - _si(54, h, w)),
                          score_scale, C_YELLOW)

    # ---- Countdown bar ----
    time_left = state["time_left"]
    cfg_secs  = state.get("round_seconds", 30.0)
    bar_y0    = h - _si(30, h, w)
    bar_y1    = h - _si(10, h, w)
    bar_xmax  = w - _si(20, h, w)
    filled    = int(bar_xmax * (time_left / max(cfg_secs, 1)))
    bar_col   = C_GREEN if time_left > 10 else (C_ORANGE if time_left > 5 else C_RED)

    _draw_panel(frame, 10, bar_y0, bar_xmax, bar_y1,
                bg=(30, 30, 30), border=C_GREY, alpha=0.6)
    cv2.rectangle(frame, (10, bar_y0), (10 + filled, bar_y1), bar_col, -1)

    time_txt = f"{time_left:.0f}s"
    time_scale = _s(0.50, h, w)
    (tw2, _), _ = cv2.getTextSize(time_txt, FONT, time_scale, 1)
    _text_with_shadow(frame, time_txt,
                      (bar_xmax - tw2 - _si(4, h, w), bar_y1 - _si(3, h, w)),
                      time_scale, C_WHITE, 1)

    # ---- Hold progress arc ----
    if phase == PLAYING and hold_progress > 0:
        cx = w - _si(54, h, w)
        cy = _si(90, h, w)
        radius = _si(34, h, w)
        end_angle = int(-90 + 360 * hold_progress)
        cv2.circle(frame, (cx, cy), radius, C_GREY, _si(3, h, w))
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, end_angle,
                    C_GREEN, _si(4, h, w))
        pct = f"{int(hold_progress * 100)}%"
        (ptw, pth), _ = cv2.getTextSize(pct, FONT, _s(0.42, h, w), 1)
        _text_with_shadow(frame, pct,
                          (cx - ptw // 2, cy + pth // 2),
                          _s(0.42, h, w), C_WHITE, 1)

    # ---- Phase flash overlays ----
    if phase == FOUND:
        flash_scale = _s(2.0, h, w)
        txt = "FOUND!"
        (ftw, _), _ = cv2.getTextSize(txt, FONT, flash_scale, 4)
        _text_with_shadow(frame, txt, (w // 2 - ftw // 2, h // 2),
                          flash_scale, C_GREEN, 4)
        last = state["history"][-1] if state["history"] else None
        if last:
            pts_txt = f"+{last.points} pts"
            if last.streak > 1:
                pts_txt += f"  (streak x{last.streak})"
            pts_scale = _s(1.0, h, w)
            (ptw2, _), _ = cv2.getTextSize(pts_txt, FONT, pts_scale, 2)
            _text_with_shadow(frame, pts_txt,
                              (w // 2 - ptw2 // 2, h // 2 + _si(54, h, w)),
                              pts_scale, C_YELLOW, 2)

    if phase == SKIP:
        skip_scale = _s(1.6, h, w)
        txt = "SKIPPED"
        (stw, _), _ = cv2.getTextSize(txt, FONT, skip_scale, 3)
        _text_with_shadow(frame, txt, (w // 2 - stw // 2, h // 2),
                          skip_scale, C_RED, 3)

    # ---- Footer hint ----
    pending = state.get("pending_settings", {})
    hint = "n=skip   s=settings   c=cam   q=quit"
    if pending:
        hint += "   (timer change next round)"
    _text_with_shadow(frame, hint, (10, h - _si(54, h, w) + _si(18, h, w)),
                      _s(0.42, h, w), C_GREY, 1)


def _draw_settings_panel(frame: np.ndarray, state: dict) -> None:
    """Overlay a semi-transparent settings panel in the top-right corner."""
    h, w = frame.shape[:2]

    panel_w = _si(300, h, w)
    panel_h = _si(140, h, w)
    x0 = w - panel_w - 10
    y0 = _si(70, h, w)

    _draw_panel(frame, x0, y0, x0 + panel_w, y0 + panel_h,
                bg=C_DARK, border=C_CYAN, alpha=0.82)

    title_scale = _s(0.55, h, w)
    row_scale   = _s(0.48, h, w)
    hint_scale  = _s(0.40, h, w)
    row_gap     = _si(28, h, w)

    _text_with_shadow(frame, "SETTINGS", (x0 + 8, y0 + _si(22, h, w)),
                      title_scale, C_CYAN, 1)

    pending = state.get("pending_settings", {})

    # Timer row
    timer_val    = state.get("round_seconds", 30.0)
    pending_timer = pending.get("round_seconds")
    timer_str = f"+/-  Timer: {timer_val:.0f}s"
    if pending_timer is not None:
        timer_str += f"  ->  {pending_timer:.0f}s (next)"
        timer_colour = C_ORANGE
    else:
        timer_colour = C_WHITE
    _text_with_shadow(frame, timer_str, (x0 + 8, y0 + _si(22, h, w) + row_gap),
                      row_scale, timer_colour, 1)

    # Confidence row
    conf_val = state.get("confidence", 0.5)
    conf_str = f"[/]  Confidence: {conf_val:.0%}"
    _text_with_shadow(frame, conf_str,
                      (x0 + 8, y0 + _si(22, h, w) + row_gap * 2),
                      row_scale, C_WHITE, 1)

    # Hint
    _text_with_shadow(frame, "s = close",
                      (x0 + 8, y0 + _si(22, h, w) + row_gap * 3),
                      hint_scale, C_GREY, 1)


def _draw_scoreboard(frame: np.ndarray, state: dict) -> None:
    h, w = frame.shape[:2]

    # Full-screen semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    title_scale = _s(1.5, h, w)
    row_scale   = _s(0.58, h, w)
    sub_scale   = _s(0.9, h, w)

    title = "GAME  OVER"
    (tw, _), _ = cv2.getTextSize(title, FONT, title_scale, 4)
    _text_with_shadow(frame, title, (w // 2 - tw // 2, _si(60, h, w)),
                      title_scale, C_YELLOW, 4)

    score_txt = f"Final score:  {state['score']}"
    (stw, _), _ = cv2.getTextSize(score_txt, FONT, sub_scale, 2)
    _text_with_shadow(frame, score_txt, (w // 2 - stw // 2, _si(108, h, w)),
                      sub_scale, C_WHITE, 2)

    # Draw a divider line
    cv2.line(frame, (w // 2 - _si(220, h, w), _si(122, h, w)),
             (w // 2 + _si(220, h, w), _si(122, h, w)), C_CYAN, 1)

    y = _si(155, h, w)
    row_h = _si(28, h, w)
    for res in state["history"]:
        icon  = "OK" if res.found else "X"
        pts   = f"+{res.points}" if res.found else "  0"
        tgt   = res.target.upper()
        time_s = f"{res.time_taken:.1f}s" if res.found else "--"
        line  = f"{res.round_number}.  {icon}  {tgt:<12}  {pts:>5} pts   {time_s}"
        colour = C_GREEN if res.found else C_RED
        (lw, _), _ = cv2.getTextSize(line, FONT, row_scale, 1)
        _text_with_shadow(frame, line, (w // 2 - lw // 2, y),
                          row_scale, colour)
        y += row_h

    quit_scale = _s(0.62, h, w)
    quit_txt   = "Press  q  to  quit"
    (qw, _), _ = cv2.getTextSize(quit_txt, FONT, quit_scale, 1)
    _text_with_shadow(frame, quit_txt, (w // 2 - qw // 2, h - _si(24, h, w)),
                      quit_scale, C_GREY)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scavenger Hunt — YOLO edition")
    p.add_argument("--camera",       type=int,   default=0,
                   help="Webcam index to start on (default 0)")
    p.add_argument("--detect-every", type=int,   default=2,
                   help="Run YOLO every N frames (default 2)")
    p.add_argument("--no-mirror",    action="store_true",
                   help="Do not horizontally flip the webcam")
    p.add_argument("--weights-dir",  type=Path,
                   default=_HERE / "weights",
                   help="Where to store model weights")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ---- Probe available cameras ----
    print("Probing cameras…")
    available_cams = _probe_cameras()
    print(f"Found cameras at indices: {available_cams}")

    # Start with the requested index if available, else first found
    cam_idx = args.camera if args.camera in available_cams else available_cams[0]
    cam_slot = available_cams.index(cam_idx)

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {cam_idx}.\n"
            "Try --camera 1 if you have more than one camera."
        )

    # ---- Model (non-fatal) ----
    print("Loading YOLO model (may download ~6 MB weights on first run)…")
    model, model_error = _load_model(args.weights_dir)
    if model_error:
        print(f"\n⚠  Model load failed:\n{model_error}\n")
        print("Running in demo mode — detection always returns empty.\n")
    else:
        print("Model ready.\n")

    # ---- Window — resizable ----
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 960, 540)

    # ---- Game ----
    game   = ScavengerHunt(Config())
    state  = game.state(time.monotonic())

    cached_detections: list[dict] = []
    frame_counter = 0

    # UI state
    settings_open = False

    # FPS tracking
    fps_prev_time = time.monotonic()
    fps           = 0.0
    fps_alpha      = 0.1  # exponential smoothing factor

    # Pulse animation (0..1 sine wave) — advances on wall-clock time, not frames
    pulse_start = time.monotonic()

    # Keep the last successfully decoded frame so the window never goes blank
    # or jitters when cap.read() returns ok=False (common on USB cameras that
    # drop a packet or aren't ready yet).
    last_good_frame: np.ndarray | None = None

    try:
        while True:
            ok, raw = cap.read()

            if ok:
                last_good_frame = raw
            elif last_good_frame is None:
                # Nothing to show yet — wait a little and retry
                time.sleep(0.01)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue
            # Always work on a copy so we don't draw on top of the stored frame
            frame = last_good_frame.copy()

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            # ---- FPS (only count real new frames) ----
            now_fps = time.monotonic()
            if ok:
                inst_fps  = 1.0 / max(now_fps - fps_prev_time, 1e-6)
                fps       = fps * (1 - fps_alpha) + inst_fps * fps_alpha
                fps_prev_time = now_fps

            # ---- Pulse animation — wall-clock so it runs even on dropped frames ----
            pulse_val = (np.sin((now_fps - pulse_start) * 3.0) + 1) / 2  # ~3 rad/s

            # ---- Detection (throttled, only on fresh frames) ----
            if ok:
                frame_counter += 1
                if frame_counter % args.detect_every == 0:
                    cached_detections = _run_detection(
                        model, frame, confidence=state.get("confidence", 0.35)
                    )

            detected_labels: list[str] = [d["label"] for d in cached_detections]

            # ---- Game tick ----
            now   = time.monotonic()
            state = game.update(detected_labels, now)

            # ---- Draw ----
            if state["phase"] == PLAYING:
                _draw_detections(frame, cached_detections, state["target"])

            hold_prog = state.get("hold_progress", 0.0)
            _draw_hud(frame, state, hold_prog, model_error,
                      fps, cam_idx, available_cams, pulse_val)

            if settings_open:
                _draw_settings_panel(frame, state)

            cv2.imshow(WIN_NAME, frame)

            # ---- Keys ----
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):            # q or ESC — quit
                break
            elif key == ord(" "):                # SPACE — start
                state = game.start(now)
            elif key == ord("n"):                # n — skip
                state = game.skip(now)
            elif key == ord("s"):                # s — toggle settings
                settings_open = not settings_open
            elif key == ord("c"):                # c — cycle camera
                if len(available_cams) > 1:
                    cam_slot = (cam_slot + 1) % len(available_cams)
                    new_idx  = available_cams[cam_slot]
                    print(f"Switching to camera {new_idx}…")
                    cap.release()
                    cap = cv2.VideoCapture(new_idx)
                    if cap.isOpened():
                        cam_idx = new_idx
                        last_good_frame = None   # don't show stale frame from old cam
                        cached_detections = []
                        print(f"Camera {cam_idx} active.")
                    else:
                        # Roll back if it failed
                        cam_slot = (cam_slot - 1) % len(available_cams)
                        cap = cv2.VideoCapture(available_cams[cam_slot])
                        print(f"Could not open camera {new_idx}, staying on {cam_idx}.")
            elif key in (ord("+"), ord("=")):    # + raise timer
                result = game.apply_settings(
                    round_seconds=game.cfg.round_seconds + 5)
                state = game.state(now)
                pending  = result.get("deferred", {})
                eff      = result.get("applied", {}).get(
                    "round_seconds",
                    result.get("deferred", {}).get("round_seconds"))
                suffix = " (next round)" if "round_seconds" in pending else ""
                print(f"Timer -> {eff:.0f}s{suffix}")
            elif key == ord("-"):                # - lower timer
                result = game.apply_settings(
                    round_seconds=game.cfg.round_seconds - 5)
                state = game.state(now)
                pending  = result.get("deferred", {})
                eff      = result.get("applied", {}).get(
                    "round_seconds",
                    result.get("deferred", {}).get("round_seconds"))
                suffix = " (next round)" if "round_seconds" in pending else ""
                print(f"Timer -> {eff:.0f}s{suffix}")
            elif key == ord("]"):                # ] raise confidence
                result = game.apply_settings(confidence=game.cfg.confidence + 0.05)
                state  = game.state(now)
                print(f"Confidence -> {result['applied'].get('confidence', game.cfg.confidence):.0%}")
            elif key == ord("["):                # [ lower confidence
                result = game.apply_settings(confidence=game.cfg.confidence - 0.05)
                state  = game.state(now)
                print(f"Confidence -> {result['applied'].get('confidence', game.cfg.confidence):.0%}")

            if state["phase"] == DONE and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
