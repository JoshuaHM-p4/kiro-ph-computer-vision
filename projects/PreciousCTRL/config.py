"""Configuration tunables for the posture checker."""

from dataclasses import dataclass, field


@dataclass
class PostureConfig:
    """All thresholds and tunables for posture detection.

    Thresholds are scale-relative (fractions of torso length or shoulder width)
    so that distance from the camera does not change behaviour.
    """

    # --- Angle thresholds (degrees) ---
    # Side view: ear-shoulder-hip angle below this is "head forward"
    head_forward_angle_bad: float = 155.0
    head_forward_angle_good: float = 162.0  # hysteresis release

    # Front view: shoulder tilt beyond this is "uneven shoulders"
    shoulder_tilt_bad: float = 8.0  # degrees from horizontal
    shoulder_tilt_good: float = 5.0  # hysteresis release

    # Side view: shoulder-hip vertical lean
    torso_lean_bad: float = 12.0  # degrees forward lean
    torso_lean_good: float = 8.0  # hysteresis release

    # --- Scoring weights (must sum to 1.0) ---
    weight_head_forward: float = 0.4
    weight_shoulder_tilt: float = 0.3
    weight_torso_lean: float = 0.3

    # --- Streak and timing ---
    # Minimum seconds of bad posture before triggering a slouch event
    slouch_trigger_seconds: float = 3.0
    # Cooldown between slouch events (avoid spamming clips)
    slouch_cooldown_seconds: float = 5.0

    # --- Replay buffer ---
    # Seconds of video to keep before slouch trigger
    replay_buffer_seconds: float = 5.0
    # Seconds of video to record after slouch trigger
    replay_after_seconds: float = 3.0
    # Max replay clips to keep on disk
    max_replay_clips: int = 50

    # --- View detection ---
    # If ear visibility drops below this, assume side view
    side_view_ear_visibility_threshold: float = 0.5
    # Minimum landmark visibility to trust a measurement
    min_landmark_visibility: float = 0.6

    # --- Display ---
    fps_target: int = 30
    window_name: str = "Posture Checker"
    replay_dir: str = "replays"
