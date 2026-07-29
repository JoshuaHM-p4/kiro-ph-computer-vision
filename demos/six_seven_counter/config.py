"""Tunables for the 6-7 rep counter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CounterConfig:
    """Settings for the see-saw rep detector.

    Counting watches the *difference* between the two wrist heights rather than
    each wrist against a fixed line. One rep is one sign flip of that difference:
    the hands swapping which one is higher. That has three advantages over
    absolute thresholds — it needs no notion of "high enough", it is unaffected by
    the whole body drifting up or down in frame, and alternation is intrinsic
    (a flip is impossible without both hands taking part).
    """

    # How far apart the wrists must be, as a fraction of torso length, before a
    # side is claimed. This deadband *is* the hysteresis: with hands roughly level
    # the last side is held, so noise cannot rack up counts.
    tilt_enter: float = 0.15
    # Smoothing on the tilt signal; lower is steadier but laggier.
    tilt_alpha: float = 0.45
    # Ignore a flip that arrives sooner than this after the previous one. Real
    # swaps take longer; a wrist jittering across the deadband does not.
    min_swap_seconds: float = 0.15

    # Shoulder-to-hip length is the preferred scale reference, but the hips are
    # often out of frame. Shoulder width times this factor stands in for it;
    # anatomically the two are close (biacromial width is a little shorter than
    # shoulder-to-hip), so the deadband keeps the same practical meaning.
    shoulder_to_torso: float = 1.2

    # Landmarks below this visibility are treated as missing.
    min_visibility: float = 0.5
    # Prepare mode: counting does not start until the camera can actually see the
    # shoulders and both wrists, held steady for this long. Without it the
    # first reps are lost while the user is still walking into frame.
    prepare_seconds: float = 1.5
    # Landmarks must sit inside the frame with this much margin to count as
    # visible; MediaPipe happily extrapolates points past the edges.
    frame_margin: float = 0.02
    # Tracking drops a frame here and there. Only fall back to prepare mode once
    # the pose has been gone this long, so a blink of lost tracking mid-rep does
    # not cost the user a fresh hold.
    lost_grace_seconds: float = 0.8

    # Rendering size for the web /snapshot canvas.
    canvas_width: int = 1280
    canvas_height: int = 720
