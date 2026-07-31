"""Game configuration — all tunables in one place."""

from dataclasses import dataclass, field


@dataclass
class GameConfig:
    """All tunables for the Don't Blink game."""

    # --- Blink detection (Eye Aspect Ratio) ---
    ear_blink_threshold: float = 0.21
    """EAR below this = eyes closed."""

    ear_open_threshold: float = 0.25
    """EAR above this = eyes open (hysteresis to prevent flicker)."""

    blink_frames: int = 2
    """Must be below threshold for this many consecutive frames to count as a blink."""

    # --- Countdown ---
    countdown_seconds: int = 3
    """3-2-1 countdown before the game starts."""

    # --- Distractions ---
    distraction_start_delay: float = 3.0
    """Seconds into the game before distractions begin."""

    distraction_interval_initial: float = 4.0
    """Seconds between distractions at the start."""

    distraction_interval_min: float = 0.5
    """Fastest distraction interval (escalates to this)."""

    distraction_interval_decay: float = 0.9
    """Multiply interval by this each time (gets faster)."""

    distraction_duration: float = 0.3
    """How long a distraction flash stays on screen (seconds)."""

    # --- Distraction types ---
    distraction_types: list[str] = field(default_factory=lambda: [
        "flash_white",       # full screen flash
        "flash_red",         # red flash
        "flash_green",       # green flash
        "shake",             # screen shake effect
        "jumpscare_text",    # scary text pops up
        "static",            # TV static
        "invert",            # invert colors
        "zoom",              # sudden zoom into your face
        "strobe",            # rapid color cycling
        "mirror",            # flip horizontally (disorienting)
        "pixelate",          # pixelate the whole frame
        "blackout",          # brief full black
        "face_warp",         # barrel distortion on face
        "split",             # split screen glitch
        "red_eye",           # turn the whole image blood red
        "tilt",              # rotate slightly
        "double_vision",     # overlay a shifted ghost of the frame
        "scanlines",         # CRT scanline effect
        "glitch_color",      # randomly swap color channels
        "shrink",            # shrink feed to center with black border
        "pulse",             # brightness pulsing
    ])

    # --- Combo mode (after certain survival time, multiple distractions stack) ---
    combo_start_time: float = 15.0
    """After this many seconds, distractions can stack (2-3 at once)."""

    combo_max_stack: int = 3
    """Maximum simultaneous distractions in combo mode."""

    # --- Near-blink panic ---
    ear_panic_threshold: float = 0.26
    """EAR below this triggers the panic visual (eyes getting tired)."""

    # --- Screen corruption (builds over time) ---
    corruption_start_time: float = 8.0
    """Seconds before the frame starts accumulating permanent corruption."""

    corruption_intensity_per_second: float = 0.01
    """How fast corruption builds (0..1 scale)."""

    # --- Taunts (shown during gameplay to distract) ---
    taunts: list[str] = field(default_factory=lambda: [
        "your eyes are drying out",
        "just blink. it'll feel so good.",
        "you WILL blink",
        "is that... a bug on your screen?",
        "don't think about blinking",
        "your eyelids are getting heavy",
        "everyone blinks. it's okay.",
        "BEHIND YOU",
        "did that just move?",
        "blink blink blink blink",
        "you're already losing",
        "wow you look tired",
        "just give up lol",
        "imagine sand in your eyes",
        "yawn... so sleepy...",
        "THE SCREEN IS WATCHING YOU",
        "how long can you really last?",
        "your body wants to blink",
        "this is unnatural",
        "fun fact: you blink 20000 times a day",
        "not blinking causes corneal damage btw",
        "i can wait. can you?",
        "tick tock tick tock",
        ":)",
        "close your eyes. rest.",
        "ERROR: human.exe not responding",
        "why are you doing this to yourself",
    ])

    # --- Penalty ---
    penalty_style: str = "bsod"
    """'bsod' for fake blue screen, 'jumpscare' for skull/scare."""

    # --- Camera ---
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # --- Face Mesh ---
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
