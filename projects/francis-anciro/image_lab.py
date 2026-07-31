"""Interactive OpenCV playground — learn the API by moving sliders.

Load an image, pick an operation with n/p keys, adjust parameters with trackbars,
and press 'r' to print the equivalent cv2 code with your current values filled in.

Usage:
    python projects/francis-anciro/image_lab.py [path/to/image.png]

If no image path is given, a test image with shapes is generated automatically.

Controls:
    n / p   — next / previous operation
    r       — print the cv2 code for the current operation and parameters
    q / Esc — quit
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np


# ==========================================================================
# Data model: operations as data, not an if/elif chain
# ==========================================================================


@dataclass(frozen=True)
class Param:
    """One tunable argument of an operation."""

    key: str
    label: str
    default: int
    minimum: int = 0
    maximum: int = 255
    odd_only: bool = False

    def coerce(self, value: int) -> int:
        """Clamp and enforce odd constraint."""
        value = max(self.minimum, min(self.maximum, int(value)))
        if self.odd_only and value % 2 == 0:
            value = max(self.minimum, value + 1)
        return value


@dataclass(frozen=True)
class Operation:
    """An OpenCV operation: knows how to run itself and describe itself as code."""

    name: str
    params: tuple[Param, ...] = field(default_factory=tuple)
    run: Callable[[np.ndarray, dict[str, int]], np.ndarray] = lambda img, p: img
    code: Callable[[dict[str, int]], list[str]] = lambda p: []
    needs_gray: bool = False

    def defaults(self) -> dict[str, int]:
        return {param.key: param.default for param in self.params}


# ==========================================================================
# Helper functions
# ==========================================================================


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert to single channel if not already."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert to 3 channels if not already."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


# ==========================================================================
# Operation implementations
# ==========================================================================


def _run_grayscale(image: np.ndarray, p: dict[str, int]) -> np.ndarray:
    return to_bgr(to_gray(image))


def _code_grayscale(p: dict[str, int]) -> list[str]:
    return [
        "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
        "img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)",
    ]


def _run_gaussian_blur(image: np.ndarray, p: dict[str, int]) -> np.ndarray:
    ksize = p["ksize"]
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def _code_gaussian_blur(p: dict[str, int]) -> list[str]:
    return [f"img = cv2.GaussianBlur(img, ({p['ksize']}, {p['ksize']}), 0)"]


def _run_median_blur(image: np.ndarray, p: dict[str, int]) -> np.ndarray:
    return cv2.medianBlur(image, p["ksize"])


def _code_median_blur(p: dict[str, int]) -> list[str]:
    return [f"img = cv2.medianBlur(img, {p['ksize']})"]


def _run_threshold(image: np.ndarray, p: dict[str, int]) -> np.ndarray:
    gray = to_gray(image)
    _, mask = cv2.threshold(gray, p["thresh"], p["maxval"], cv2.THRESH_BINARY)
    return to_bgr(mask)


def _code_threshold(p: dict[str, int]) -> list[str]:
    return [
        "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
        f"_, mask = cv2.threshold(gray, {p['thresh']}, {p['maxval']}, cv2.THRESH_BINARY)",
        "img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)",
    ]


def _run_adaptive_threshold(image: np.ndarray, p: dict[str, int]) -> np.ndarray:
    gray = to_gray(image)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, p["block"], p["c"]
    )
    return to_bgr(mask)


def _code_adaptive_threshold(p: dict[str, int]) -> list[str]:
    return [
        "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
        "mask = cv2.adaptiveThreshold(",
        f"    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,",
        f"    cv2.THRESH_BINARY, {p['block']}, {p['c']},",
        ")",
        "img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)",
    ]


def _run_canny(image: np.ndarray, p: dict[str, int]) -> np.ndarray:
    gray = to_gray(image)
    edges = cv2.Canny(gray, p["low"], p["high"], apertureSize=p["aperture"])
    return to_bgr(edges)


def _code_canny(p: dict[str, int]) -> list[str]:
    return [
        "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
        f"edges = cv2.Canny(gray, {p['low']}, {p['high']}, apertureSize={p['aperture']})",
        "img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)",
    ]


def _run_sobel(image: np.ndarray, p: dict[str, int]) -> np.ndarray:
    gray = to_gray(image)
    ksize = p["ksize"]
    dx = p["dx"]
    dy = p["dy"]
    # At least one must be non-zero
    if dx == 0 and dy == 0:
        dx = 1
    grad = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
    return to_bgr(cv2.convertScaleAbs(grad))


def _code_sobel(p: dict[str, int]) -> list[str]:
    dx, dy = p["dx"], p["dy"]
    if dx == 0 and dy == 0:
        dx = 1
    return [
        "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
        f"grad = cv2.Sobel(gray, cv2.CV_64F, {dx}, {dy}, ksize={p['ksize']})",
        "img = cv2.cvtColor(cv2.convertScaleAbs(grad), cv2.COLOR_GRAY2BGR)",
    ]


_MORPH_OPS = {
    0: ("erode", cv2.MORPH_ERODE),
    1: ("dilate", cv2.MORPH_DILATE),
    2: ("open", cv2.MORPH_OPEN),
    3: ("close", cv2.MORPH_CLOSE),
}


def _run_morphology(image: np.ndarray, p: dict[str, int]) -> np.ndarray:
    ksize = p["ksize"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    op_index = p["op"]
    _, morph_op = _MORPH_OPS.get(op_index, ("erode", cv2.MORPH_ERODE))
    iters = p["iterations"]
    if morph_op == cv2.MORPH_ERODE:
        return cv2.erode(image, kernel, iterations=iters)
    elif morph_op == cv2.MORPH_DILATE:
        return cv2.dilate(image, kernel, iterations=iters)
    else:
        return cv2.morphologyEx(image, morph_op, kernel, iterations=iters)


def _code_morphology(p: dict[str, int]) -> list[str]:
    ksize = p["ksize"]
    op_index = p["op"]
    op_name, _ = _MORPH_OPS.get(op_index, ("erode", cv2.MORPH_ERODE))
    iters = p["iterations"]
    lines = [
        f"kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({ksize}, {ksize}))",
    ]
    if op_name == "erode":
        lines.append(f"img = cv2.erode(img, kernel, iterations={iters})")
    elif op_name == "dilate":
        lines.append(f"img = cv2.dilate(img, kernel, iterations={iters})")
    else:
        lines.append(
            f"img = cv2.morphologyEx(img, cv2.MORPH_{op_name.upper()}, kernel, iterations={iters})"
        )
    return lines


def _run_resize(image: np.ndarray, p: dict[str, int]) -> np.ndarray:
    scale = p["scale_pct"] / 100.0
    if scale <= 0:
        scale = 0.1
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _code_resize(p: dict[str, int]) -> list[str]:
    scale = p["scale_pct"] / 100.0
    if scale <= 0:
        scale = 0.1
    return [
        f"img = cv2.resize(img, None, fx={scale:.2f}, fy={scale:.2f}, interpolation=cv2.INTER_AREA)"
    ]


def _run_rotate(image: np.ndarray, p: dict[str, int]) -> np.ndarray:
    angle = p["angle"] - 180  # slider 0..360 maps to -180..180
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def _code_rotate(p: dict[str, int]) -> list[str]:
    angle = p["angle"] - 180
    return [
        "h, w = img.shape[:2]",
        f"matrix = cv2.getRotationMatrix2D((w / 2, h / 2), {angle}, 1.0)",
        "img = cv2.warpAffine(img, matrix, (w, h))",
    ]


def _run_contours(image: np.ndarray, p: dict[str, int]) -> np.ndarray:
    gray = to_gray(image)
    _, mask = cv2.threshold(gray, p["thresh"], 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = p["min_area"]
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    output = to_bgr(image).copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    for c in contours:
        # Bounding box
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 200, 0), 2)
        # Centroid
        m = cv2.moments(c)
        if m["m00"]:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

    return output


def _code_contours(p: dict[str, int]) -> list[str]:
    return [
        "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
        f"_, mask = cv2.threshold(gray, {p['thresh']}, 255, cv2.THRESH_BINARY)",
        "contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)",
        f"contours = [c for c in contours if cv2.contourArea(c) >= {p['min_area']}]",
        "cv2.drawContours(img, contours, -1, (0, 255, 0), 2)",
        "for c in contours:",
        "    x, y, w, h = cv2.boundingRect(c)",
        "    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 0), 2)",
        "    m = cv2.moments(c)",
        "    if m['m00']:",
        "        cx, cy = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])",
        "        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)",
    ]


# ==========================================================================
# The operations table — adding an operation means adding one entry here
# ==========================================================================

OPERATIONS: tuple[Operation, ...] = (
    Operation(
        name="Grayscale",
        run=_run_grayscale,
        code=_code_grayscale,
    ),
    Operation(
        name="Gaussian Blur",
        params=(
            Param("ksize", "Kernel Size", default=5, minimum=1, maximum=61, odd_only=True),
        ),
        run=_run_gaussian_blur,
        code=_code_gaussian_blur,
    ),
    Operation(
        name="Median Blur",
        params=(
            Param("ksize", "Kernel Size", default=5, minimum=1, maximum=61, odd_only=True),
        ),
        run=_run_median_blur,
        code=_code_median_blur,
    ),
    Operation(
        name="Threshold (Fixed)",
        params=(
            Param("thresh", "Threshold", default=127, minimum=0, maximum=255),
            Param("maxval", "Max Value", default=255, minimum=0, maximum=255),
        ),
        needs_gray=True,
        run=_run_threshold,
        code=_code_threshold,
    ),
    Operation(
        name="Adaptive Threshold",
        params=(
            Param("block", "Block Size", default=11, minimum=3, maximum=61, odd_only=True),
            Param("c", "Constant C", default=5, minimum=0, maximum=40),
        ),
        needs_gray=True,
        run=_run_adaptive_threshold,
        code=_code_adaptive_threshold,
    ),
    Operation(
        name="Canny Edges",
        params=(
            Param("low", "Lower Thresh", default=80, minimum=0, maximum=500),
            Param("high", "Upper Thresh", default=180, minimum=0, maximum=500),
            Param("aperture", "Aperture", default=3, minimum=3, maximum=7, odd_only=True),
        ),
        needs_gray=True,
        run=_run_canny,
        code=_code_canny,
    ),
    Operation(
        name="Sobel Gradient",
        params=(
            Param("dx", "dx (0 or 1)", default=1, minimum=0, maximum=1),
            Param("dy", "dy (0 or 1)", default=0, minimum=0, maximum=1),
            Param("ksize", "Kernel Size", default=3, minimum=1, maximum=7, odd_only=True),
        ),
        needs_gray=True,
        run=_run_sobel,
        code=_code_sobel,
    ),
    Operation(
        name="Morphology",
        params=(
            Param("op", "Op (0=erode 1=dilate 2=open 3=close)", default=0, minimum=0, maximum=3),
            Param("ksize", "Kernel Size", default=5, minimum=1, maximum=31, odd_only=True),
            Param("iterations", "Iterations", default=1, minimum=1, maximum=10),
        ),
        run=_run_morphology,
        code=_code_morphology,
    ),
    Operation(
        name="Resize",
        params=(
            Param("scale_pct", "Scale %", default=50, minimum=10, maximum=200),
        ),
        run=_run_resize,
        code=_code_resize,
    ),
    Operation(
        name="Rotate",
        params=(
            Param("angle", "Angle (+180 offset)", default=195, minimum=0, maximum=360),
        ),
        run=_run_rotate,
        code=_code_rotate,
    ),
    Operation(
        name="Find Contours",
        params=(
            Param("thresh", "Threshold", default=127, minimum=0, maximum=255),
            Param("min_area", "Min Area (px)", default=200, minimum=0, maximum=20000),
        ),
        needs_gray=True,
        run=_run_contours,
        code=_code_contours,
    ),
)


# ==========================================================================
# Test image generator
# ==========================================================================


def generate_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a colourful test image with shapes useful for every operation."""
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Gradient background (interesting for blurs and thresholds)
    for y in range(height):
        intensity = int(255 * y / height)
        image[y, :] = (intensity // 2, intensity, 255 - intensity)

    # Add noise band (useful for blur demos)
    noise_region = image[10:60, :].copy()
    noise = np.random.randint(0, 80, noise_region.shape, dtype=np.uint8)
    image[10:60, :] = cv2.add(noise_region, noise)

    # Shapes for contour detection
    cv2.rectangle(image, (50, 100), (180, 230), (255, 255, 255), -1)
    cv2.circle(image, (320, 180), 70, (0, 200, 255), -1)
    cv2.ellipse(image, (500, 150), (80, 50), 30, 0, 360, (200, 100, 255), -1)

    # Triangle
    pts = np.array([[400, 350], [480, 450], [320, 450]], np.int32)
    cv2.fillPoly(image, [pts], (100, 255, 100))

    # Small circles (test min_area filtering)
    for x in range(220, 420, 30):
        cv2.circle(image, (x, 400), 8, (255, 255, 255), -1)

    # Text
    cv2.putText(
        image, "OpenCV Image Lab", (150, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
    )

    return image


# ==========================================================================
# UI: trackbar window and main loop
# ==========================================================================

WINDOW_NAME = "Image Lab"
TRACKBAR_WINDOW = "Parameters"


def nothing(x: Any) -> None:
    """No-op callback for trackbars."""
    pass


def setup_trackbars(operation: Operation) -> None:
    """Destroy and recreate the parameter window with trackbars for this operation."""
    try:
        cv2.destroyWindow(TRACKBAR_WINDOW)
    except cv2.error:
        pass  # Window doesn't exist yet on first call
    cv2.namedWindow(TRACKBAR_WINDOW, cv2.WINDOW_AUTOSIZE)
    for param in operation.params:
        cv2.createTrackbar(param.label, TRACKBAR_WINDOW, param.default, param.maximum, nothing)


def read_trackbars(operation: Operation) -> dict[str, int]:
    """Read current trackbar positions and coerce them."""
    values: dict[str, int] = {}
    for param in operation.params:
        try:
            raw = cv2.getTrackbarPos(param.label, TRACKBAR_WINDOW)
        except cv2.error:
            raw = param.default
        values[param.key] = param.coerce(raw)
    return values


def print_code(operation: Operation, params: dict[str, int]) -> None:
    """Print the cv2 code that reproduces the current preview."""
    print("\n" + "=" * 60)
    print(f"# Operation: {operation.name}")
    print("# Generated code (copy-paste ready):")
    print("-" * 60)
    lines = operation.code(params)
    for line in lines:
        print(line)
    print("=" * 60 + "\n")


def make_side_by_side(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """Place original and processed images side by side at uniform height."""
    target_h = original.shape[0]
    target_w = original.shape[1]

    # Resize processed to match original height, preserving aspect ratio
    proc_bgr = to_bgr(processed)
    ph, pw = proc_bgr.shape[:2]
    if ph != target_h:
        scale = target_h / ph
        new_w = max(1, int(pw * scale))
        proc_bgr = cv2.resize(proc_bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)

    # Pad processed width to match original if narrower, or just concat
    orig_bgr = to_bgr(original)

    # Separator bar
    sep = np.full((target_h, 3, 3), (80, 80, 80), dtype=np.uint8)

    return np.hstack([orig_bgr, sep, proc_bgr])


def main() -> None:
    # Load or generate image
    if len(sys.argv) > 1:
        path = sys.argv[1]
        image = cv2.imread(path)
        if image is None:
            print(f"Error: could not load image '{path}'")
            sys.exit(1)
        print(f"Loaded: {path} ({image.shape[1]}x{image.shape[0]})")
    else:
        image = generate_test_image()
        print("No image path given — using generated test image.")

    # Resize if too large for comfortable viewing
    max_side = 600
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    current_index = 0
    num_ops = len(OPERATIONS)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    setup_trackbars(OPERATIONS[current_index])

    print("\nControls:")
    print("  n / p  — next / previous operation")
    print("  r      — print equivalent cv2 code")
    print("  q / Esc — quit")
    print(f"\nCurrent operation: {OPERATIONS[current_index].name}")

    while True:
        operation = OPERATIONS[current_index]
        params = read_trackbars(operation)

        # Apply operation (with error recovery)
        try:
            processed = operation.run(image, params)
        except cv2.error:
            # Skip failing parameter combos — keep showing original
            processed = image.copy()

        # HUD: show operation name on the processed side
        display = make_side_by_side(image, processed)
        label = f"[{current_index + 1}/{num_ops}] {operation.name}"
        cv2.putText(
            display, label, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA,
        )
        cv2.putText(
            display, "n/p: switch  r: code  q: quit", (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("n"):
            current_index = (current_index + 1) % num_ops
            setup_trackbars(OPERATIONS[current_index])
            print(f"Operation: {OPERATIONS[current_index].name}")
        elif key == ord("p"):
            current_index = (current_index - 1) % num_ops
            setup_trackbars(OPERATIONS[current_index])
            print(f"Operation: {OPERATIONS[current_index].name}")
        elif key == ord("r"):
            print_code(operation, params)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
