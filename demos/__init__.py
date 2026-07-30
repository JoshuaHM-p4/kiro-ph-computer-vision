"""OpenCV + MediaPipe demo suite.

Four interactive computer-vision demos, each available as a desktop OpenCV app
and as a Flask web app:

    air_canvas         paint in the air with hand gestures
    slide_presenter    laser pointer + pinch-to-advance slide control
    six_seven_counter  alternating-hand "6-7" rep counter from pose
    pngtuber           head-yaw and expression driven sprite switcher

Every demo keeps its logic in a pure ``core`` module that consumes normalized
landmarks and returns state, so the desktop loop and the WebSocket handler are
both thin adapters over one implementation.
"""

__all__ = ["DEMOS", "DemoInfo"]

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoInfo:
    """Registry entry describing one demo."""

    slug: str
    title: str
    tagline: str
    description: str
    port: int
    module: str

    @property
    def desktop_module(self) -> str:
        return f"{self.module}.desktop"

    @property
    def web_module(self) -> str:
        return f"{self.module}.web"


DEMOS: tuple[DemoInfo, ...] = (
    DemoInfo(
        slug="image-lab",
        title="Image Lab",
        tagline="OpenCV, hands on",
        description=(
            "Upload an image or grab a webcam frame, then chain OpenCV operations - blurs, "
            "thresholds, edges, morphology, contours, shapes and text - and reveal the exact "
            "cv2 code for the settings you picked."
        ),
        port=5005,
        module="demos.image_lab",
    ),
    DemoInfo(
        slug="air-canvas",
        title="Air Canvas",
        tagline="Paint with your fingertip",
        description=(
            "Face mesh and hand keypoints with a gesture-driven paint engine: "
            "draw, erase, and pick color, size and opacity from a dwell-activated palette."
        ),
        port=5001,
        module="demos.air_canvas",
    ),
    DemoInfo(
        slug="slide-presenter",
        title="Slide Presenter",
        tagline="Pinch to change slides",
        description=(
            "Present a folder of slide images. Your index finger is a laser pointer; "
            "pinch with the right hand for the next slide and the left hand to go back."
        ),
        port=5002,
        module="demos.slide_presenter",
    ),
    DemoInfo(
        slug="six-seven",
        title="6-7 Counter",
        tagline="Alternating hand reps",
        description=(
            "Pose estimation counts 6-7 reps as hand swaps: every time your hands trade "
            "which one is higher. Nothing to calibrate, and one hand alone cannot score."
        ),
        port=5003,
        module="demos.six_seven_counter",
    ),
    DemoInfo(
        slug="pngtuber",
        title="PNGTuber",
        tagline="Yaw and expression sprites",
        description=(
            "Head yaw from Face Mesh picks a left/center/right sprite, and mouth, brow "
            "and eye ratios pick the expression: neutral, happy, surprised or angry."
        ),
        port=5004,
        module="demos.pngtuber",
    ),
    DemoInfo(
        slug="scavenger-hunt",
        title="Scavenger Hunt",
        tagline="Bring me a cup, fast",
        description=(
            "A game on top of a COCO-pretrained YOLO model: 'bring me a cell phone' appears "
            "with a 30 second timer, and you hold the real object up to the webcam or upload "
            "a photo of it. Needs a user-provided model."
        ),
        port=5006,
        module="demos.scavenger_hunt",
    ),
    DemoInfo(
        slug="sam-labeler",
        title="SAM Labeler",
        tagline="Segment anything you can name",
        description=(
            "Capture or upload a picture, type what to look for in plain words, and SAM 3.1 "
            "returns masks. Then style each label with OpenCV: fill, outline, blur, pixelate, "
            "spotlight or cutout. Needs a Hugging Face token, or run it in demo mode."
        ),
        port=5007,
        module="demos.sam_labeler",
    ),
)
