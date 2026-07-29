"""Shared "futuristic" HUD theme.

All helpers take and mutate a BGR numpy frame in place and return it, so calls
chain. Colors are BGR tuples because that is what OpenCV expects.

The glow effect is a blur-and-add: draw the shape onto a black scratch layer,
blur it, then screen it back over the frame. That reads as neon bloom while
staying a couple of cheap numpy ops per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np

from . import landmarks as lm
from .geometry import clamp, to_pixels

FONT = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_PLAIN


@dataclass(frozen=True)
class Theme:
    """Palette for the HUD. BGR tuples."""

    cyan: tuple[int, int, int] = (255, 231, 76)
    magenta: tuple[int, int, int] = (200, 60, 255)
    lime: tuple[int, int, int] = (120, 255, 140)
    amber: tuple[int, int, int] = (60, 190, 255)
    red: tuple[int, int, int] = (80, 80, 255)
    white: tuple[int, int, int] = (245, 245, 245)
    dim: tuple[int, int, int] = (140, 140, 140)
    panel: tuple[int, int, int] = (18, 14, 26)
    grid: tuple[int, int, int] = (60, 40, 80)


THEME = Theme()


def scanlines(frame: np.ndarray, spacing: int = 3, strength: float = 0.12) -> np.ndarray:
    """Darken every ``spacing``-th row for a CRT feel."""
    if strength <= 0:
        return frame
    frame[::spacing] = (frame[::spacing] * (1.0 - strength)).astype(frame.dtype)
    return frame


def vignette(frame: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Darken the frame edges to pull the eye toward the centre."""
    height, width = frame.shape[:2]
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radius = np.sqrt(x * x + y * y) / np.sqrt(2.0)
    mask = 1.0 - strength * radius
    frame[:] = np.clip(frame.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)
    return frame


def glow(
    frame: np.ndarray,
    draw_fn,
    *,
    blur: int = 25,
    intensity: float = 0.85,
) -> np.ndarray:
    """Composite a neon bloom for whatever ``draw_fn`` paints.

    ``draw_fn`` receives a black scratch layer the same size as ``frame`` and
    draws the shape to be glowed. The layer is blurred and added, then the crisp
    shape is drawn straight onto the frame on top.
    """
    layer = np.zeros_like(frame)
    draw_fn(layer)
    blur = max(3, blur | 1)  # GaussianBlur needs an odd kernel
    bloom = cv2.GaussianBlur(layer, (blur, blur), 0)
    cv2.addWeighted(frame, 1.0, bloom, intensity, 0, dst=frame)
    draw_fn(frame)
    return frame


def panel(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    *,
    alpha: float = 0.55,
    color: tuple[int, int, int] | None = None,
    border: tuple[int, int, int] | None = None,
    corner: int = 14,
) -> np.ndarray:
    """Translucent panel with bracketed corners."""
    color = color or THEME.panel
    border = border or THEME.cyan
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1, x2 = max(0, min(x1, x2)), min(frame.shape[1], max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(frame.shape[0], max(y1, y2))
    if x2 <= x1 or y2 <= y1:
        return frame

    region = frame[y1:y2, x1:x2]
    fill = np.full_like(region, color, dtype=np.uint8)
    cv2.addWeighted(fill, alpha, region, 1.0 - alpha, 0, dst=region)

    for (cx, cy, dx, dy) in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        cv2.line(frame, (cx, cy), (cx + dx * corner, cy), border, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner), border, 2, cv2.LINE_AA)
    return frame


def text(
    frame: np.ndarray,
    label: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.5,
    color: tuple[int, int, int] | None = None,
    thickness: int = 1,
    shadow: bool = True,
) -> np.ndarray:
    """Draw text with a soft drop shadow so it survives busy backgrounds."""
    color = color or THEME.white
    if shadow:
        cv2.putText(frame, label, (origin[0] + 1, origin[1] + 1), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, label, origin, FONT, scale, color, thickness, cv2.LINE_AA)
    return frame


def title(frame: np.ndarray, label: str, subtitle: str = "") -> np.ndarray:
    """Top-left title block used by every demo."""
    panel(frame, (12, 10), (12 + 18 * len(label) // 2 + 190, 60 if subtitle else 44))
    text(frame, label, (26, 36), scale=0.72, color=THEME.cyan, thickness=2)
    if subtitle:
        text(frame, subtitle, (26, 54), scale=0.42, color=THEME.dim)
    return frame


def status_strip(
    frame: np.ndarray,
    entries: Sequence[tuple[str, str]],
    *,
    y: int | None = None,
) -> np.ndarray:
    """Bottom strip of ``LABEL value`` pairs."""
    height, width = frame.shape[:2]
    y = height - 34 if y is None else y
    panel(frame, (12, y), (width - 12, y + 26), alpha=0.6)
    x = 26
    for label, value in entries:
        text(frame, label, (x, y + 18), scale=0.4, color=THEME.dim)
        x += 9 * len(label) + 6
        text(frame, value, (x, y + 18), scale=0.44, color=THEME.cyan, thickness=1)
        x += 9 * len(value) + 22
    return frame


def gauge(
    frame: np.ndarray,
    top_left: tuple[int, int],
    size: tuple[int, int],
    value: float,
    *,
    label: str = "",
    color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Horizontal bar gauge for a 0..1 value."""
    color = color or THEME.cyan
    x, y = top_left
    w, h = size
    value = clamp(value, 0.0, 1.0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), THEME.grid, 1, cv2.LINE_AA)
    filled = int(w * value)
    if filled > 1:
        cv2.rectangle(frame, (x + 1, y + 1), (x + filled - 1, y + h - 1), color, -1)
    if label:
        text(frame, label, (x, y - 6), scale=0.38, color=THEME.dim)
    return frame


def ring(
    frame: np.ndarray,
    center: tuple[int, int],
    radius: int,
    progress: float,
    *,
    color: tuple[int, int, int] | None = None,
    thickness: int = 3,
) -> np.ndarray:
    """Circular progress ring, used for dwell feedback."""
    color = color or THEME.magenta
    cv2.circle(frame, center, radius, THEME.grid, 1, cv2.LINE_AA)
    if progress > 0:
        end = int(360 * clamp(progress, 0.0, 1.0))
        cv2.ellipse(frame, center, (radius, radius), -90, 0, end, color, thickness, cv2.LINE_AA)
    return frame


def crosshair(
    frame: np.ndarray,
    center: tuple[int, int],
    *,
    radius: int = 16,
    color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Reticle marking the active pointer."""
    color = color or THEME.lime
    x, y = center
    cv2.circle(frame, center, radius, color, 1, cv2.LINE_AA)
    cv2.line(frame, (x - radius - 6, y), (x - 4, y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x + 4, y), (x + radius + 6, y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y - radius - 6), (x, y - 4), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y + 4), (x, y + radius + 6), color, 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Landmark overlays
# ---------------------------------------------------------------------------


def draw_hand(
    frame: np.ndarray,
    hand: lm.Hand,
    *,
    color: tuple[int, int, int] | None = None,
    joint_color: tuple[int, int, int] | None = None,
    show_label: bool = True,
) -> np.ndarray:
    """Hand skeleton with joint dots."""
    color = color or THEME.cyan
    joint_color = joint_color or THEME.white
    height, width = frame.shape[:2]
    points = [to_pixels(p, width, height) for p in hand.points]
    for start, end in lm.HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        radius = 5 if index in lm.FINGER_TIPS else 3
        cv2.circle(frame, point, radius, joint_color, -1, cv2.LINE_AA)
    if show_label and hand.label != "Unknown":
        text(frame, hand.label.upper(), (points[lm.WRIST][0] - 20, points[lm.WRIST][1] + 22), scale=0.42, color=color)
    return frame


def _polyline(frame: np.ndarray, points: Sequence[tuple[int, int]], color, thickness=1, closed=True) -> None:
    if len(points) < 2:
        return
    cv2.polylines(frame, [np.array(points, dtype=np.int32)], closed, color, thickness, cv2.LINE_AA)


def draw_face(
    frame: np.ndarray,
    face: lm.Face,
    *,
    color: tuple[int, int, int] | None = None,
    dense: bool = False,
) -> np.ndarray:
    """Face contours, or the full point cloud when ``dense`` is set."""
    color = color or THEME.magenta
    height, width = frame.shape[:2]

    if dense:
        for point in face.points:
            cv2.circle(frame, to_pixels(point, width, height), 1, color, -1, cv2.LINE_AA)
        return frame

    for contour in (
        lm.FACE_OVAL,
        lm.LEFT_EYE_RING,
        lm.RIGHT_EYE_RING,
        lm.OUTER_LIPS,
        lm.INNER_LIPS,
    ):
        _polyline(frame, [to_pixels(face.points[i], width, height) for i in contour], color)
    for brow in (lm.LEFT_BROW, lm.RIGHT_BROW):
        _polyline(frame, [to_pixels(face.points[i], width, height) for i in brow], color, closed=False)
    return frame


def draw_pose(
    frame: np.ndarray,
    pose: lm.Pose,
    *,
    color: tuple[int, int, int] | None = None,
    joint_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Upper-body pose skeleton."""
    color = color or THEME.lime
    joint_color = joint_color or THEME.white
    height, width = frame.shape[:2]
    points = [to_pixels(p, width, height) for p in pose.points]
    for start, end in lm.POSE_CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 3, cv2.LINE_AA)
    for name in lm.POSE:
        cv2.circle(frame, points[lm.POSE[name]], 5, joint_color, -1, cv2.LINE_AA)
    return frame


def draw_frame_landmarks(
    frame: np.ndarray,
    landmark_frame: lm.LandmarkFrame,
    *,
    hands: bool = True,
    face: bool = True,
    pose: bool = True,
) -> np.ndarray:
    """Draw whatever the frame contains, respecting the enable flags."""
    if pose and landmark_frame.pose is not None:
        draw_pose(frame, landmark_frame.pose)
    if face and landmark_frame.face is not None:
        draw_face(frame, landmark_frame.face)
    if hands:
        for hand in landmark_frame.hands:
            draw_hand(frame, hand)
    return frame


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def alpha_blit(background: np.ndarray, sprite_bgra: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    """Alpha-composite a BGRA sprite onto a BGR frame, clipped at the edges."""
    if sprite_bgra.ndim != 3 or sprite_bgra.shape[2] != 4:
        raise ValueError("sprite must be BGRA")

    bh, bw = background.shape[:2]
    sh, sw = sprite_bgra.shape[:2]
    x, y = top_left

    # Intersect the sprite rect with the frame so off-screen sprites clip.
    sx1, sy1 = max(0, -x), max(0, -y)
    dx1, dy1 = max(0, x), max(0, y)
    dx2, dy2 = min(bw, x + sw), min(bh, y + sh)
    if dx2 <= dx1 or dy2 <= dy1:
        return background

    sprite = sprite_bgra[sy1 : sy1 + (dy2 - dy1), sx1 : sx1 + (dx2 - dx1)]
    alpha = (sprite[:, :, 3:4].astype(np.float32) / 255.0)
    region = background[dy1:dy2, dx1:dx2].astype(np.float32)
    blended = sprite[:, :, :3].astype(np.float32) * alpha + region * (1.0 - alpha)
    background[dy1:dy2, dx1:dx2] = blended.astype(np.uint8)
    return background


def draw_strokes_rgba(
    layer_bgra: np.ndarray,
    strokes: Iterable["StrokeLike"],
) -> np.ndarray:
    """Rasterize strokes onto a BGRA layer, each with its own alpha.

    Each stroke is drawn on its own scratch layer so overlapping segments within
    one stroke do not stack alpha, then merged with the standard source-over
    formula. That keeps a 40% opacity stroke reading as 40% no matter how many
    times it crosses itself.
    """
    for stroke in strokes:
        points = getattr(stroke, "pixels", None) or []
        if len(points) == 0:
            continue
        scratch = np.zeros(layer_bgra.shape[:2], dtype=np.uint8)
        thickness = max(1, int(getattr(stroke, "size", 8)))
        if len(points) == 1:
            cv2.circle(scratch, points[0], max(1, thickness // 2), 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(scratch, [np.array(points, dtype=np.int32)], False, 255, thickness, cv2.LINE_AA)

        opacity = float(getattr(stroke, "opacity", 1.0))
        if getattr(stroke, "erase", False):
            keep = 1.0 - (scratch.astype(np.float32) / 255.0) * opacity
            layer_bgra[:, :, 3] = (layer_bgra[:, :, 3].astype(np.float32) * keep).astype(np.uint8)
            continue

        src_a = (scratch.astype(np.float32) / 255.0) * opacity
        dst_a = layer_bgra[:, :, 3].astype(np.float32) / 255.0
        out_a = src_a + dst_a * (1.0 - src_a)
        color = np.array(getattr(stroke, "color", (255, 255, 255)), dtype=np.float32)
        src_rgb = np.broadcast_to(color, (*scratch.shape, 3))
        dst_rgb = layer_bgra[:, :, :3].astype(np.float32)
        safe_a = np.where(out_a > 1e-6, out_a, 1.0)[:, :, None]
        out_rgb = (
            src_rgb * src_a[:, :, None] + dst_rgb * dst_a[:, :, None] * (1.0 - src_a[:, :, None])
        ) / safe_a
        layer_bgra[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        layer_bgra[:, :, 3] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)
    return layer_bgra


class StrokeLike:  # pragma: no cover - structural documentation only
    """Duck type expected by :func:`draw_strokes_rgba`."""

    pixels: list[tuple[int, int]]
    color: tuple[int, int, int]
    size: int
    opacity: float
    erase: bool
