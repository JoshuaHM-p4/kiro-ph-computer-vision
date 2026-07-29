"""Catalog of OpenCV operations the image lab can apply.

Each entry knows three things: how to **run** the operation, what **parameters** it
takes, and the **cv2 code** that reproduces it. Keeping those together is the whole
point of the demo — the UI is a thin skin over this table, and the "reveal code"
button just asks every step in the pipeline to describe itself.

Adding an operation means adding one :class:`Operation` here; nothing else needs to
change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

# Categories, in the order the UI should present them.
CATEGORIES: tuple[str, ...] = (
    "Color",
    "Blur",
    "Threshold",
    "Edges",
    "Morphology",
    "Geometry",
    "Features",
    "Drawing",
)


@dataclass(frozen=True)
class Param:
    """One tunable argument of an operation."""

    key: str
    label: str
    kind: str = "int"  # int | float | bool | choice | color
    default: Any = 0
    minimum: float = 0.0
    maximum: float = 100.0
    step: float = 1.0
    choices: tuple[str, ...] = ()
    # Kernel sizes must be odd and >= 1; the UI and the runner both enforce it.
    odd_only: bool = False
    help: str = ""

    def coerce(self, value: Any) -> Any:
        """Clamp and type-correct a value coming from a UI or a JSON payload."""
        if self.kind == "bool":
            return bool(value)
        if self.kind == "choice":
            return value if value in self.choices else self.default
        if self.kind == "color":
            try:
                b, g, r = (int(channel) for channel in value)
            except (TypeError, ValueError):
                return self.default
            return (max(0, min(255, b)), max(0, min(255, g)), max(0, min(255, r)))
        try:
            number = float(value)
        except (TypeError, ValueError):
            return self.default
        number = max(self.minimum, min(self.maximum, number))
        if self.kind == "int":
            number = int(round(number))
            if self.odd_only and number % 2 == 0:
                number += 1  # 4 -> 5: GaussianBlur rejects even kernels
            return number
        return round(number, 4)

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "default": list(self.default) if self.kind == "color" else self.default,
            "min": self.minimum,
            "max": self.maximum,
            "step": self.step,
            "choices": list(self.choices),
            "oddOnly": self.odd_only,
            "help": self.help,
        }


@dataclass(frozen=True)
class Operation:
    """An OpenCV call, its parameters, and the code that reproduces it."""

    key: str
    label: str
    category: str
    summary: str
    run: Callable[[np.ndarray, dict[str, Any]], np.ndarray]
    code: Callable[[dict[str, Any]], list[str]]
    params: tuple[Param, ...] = field(default_factory=tuple)
    # Operations that need a single-channel input; the runner converts first and
    # the generated code shows that conversion, so the snippet always runs.
    needs_gray: bool = False
    docs: str = ""

    def defaults(self) -> dict[str, Any]:
        return {param.key: param.default for param in self.params}

    def coerce(self, values: dict[str, Any] | None) -> dict[str, Any]:
        values = values or {}
        return {
            param.key: param.coerce(values.get(param.key, param.default))
            for param in self.params
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "category": self.category,
            "summary": self.summary,
            "needsGray": self.needs_gray,
            "docs": self.docs,
            "params": [param.to_json() for param in self.params],
        }


# --------------------------------------------------------------------------
# Helpers shared by the operations
# --------------------------------------------------------------------------


def to_gray(image: np.ndarray) -> np.ndarray:
    """Single channel view of an image, whatever it arrived as."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Three channel view, so every operation can be chained after any other."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _kernel(size: int, shape: str = "rect") -> np.ndarray:
    shapes = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    return cv2.getStructuringElement(shapes.get(shape, cv2.MORPH_RECT), (size, size))


def _point(image: np.ndarray, fx: float, fy: float) -> tuple[int, int]:
    """Fractional coordinates to pixels, so drawing survives a resize."""
    height, width = image.shape[:2]
    return int(fx * width), int(fy * height)


# --------------------------------------------------------------------------
# Color
# --------------------------------------------------------------------------

_OPERATIONS: list[Operation] = []


def register(operation: Operation) -> Operation:
    _OPERATIONS.append(operation)
    return operation


register(
    Operation(
        key="grayscale",
        label="Grayscale",
        category="Color",
        summary="Collapse the three colour channels into one intensity channel.",
        docs="cvtColor",
        run=lambda image, p: to_bgr(to_gray(image)),
        code=lambda p: [
            "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
            "img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # back to 3 channels to keep chaining",
        ],
    )
)

register(
    Operation(
        key="brightness_contrast",
        label="Brightness / Contrast",
        category="Color",
        summary="Linear intensity rescale: out = alpha * in + beta.",
        docs="convertScaleAbs",
        params=(
            Param("alpha", "Contrast (alpha)", "float", 1.2, 0.1, 3.0, 0.05),
            Param("beta", "Brightness (beta)", "int", 10, -100, 100, 1),
        ),
        run=lambda image, p: cv2.convertScaleAbs(image, alpha=p["alpha"], beta=p["beta"]),
        code=lambda p: [f"img = cv2.convertScaleAbs(img, alpha={p['alpha']}, beta={p['beta']})"],
    )
)

register(
    Operation(
        key="hue_shift",
        label="Hue shift",
        category="Color",
        summary="Rotate hue in HSV space, leaving saturation and value alone.",
        docs="cvtColor + split",
        params=(Param("shift", "Hue shift", "int", 40, 0, 179, 1),),
        run=lambda image, p: _hue_shift(image, p["shift"]),
        code=lambda p: [
            "hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)",
            "h, s, v = cv2.split(hsv)",
            f"h = ((h.astype(int) + {p['shift']}) % 180).astype('uint8')",
            "img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)",
        ],
    )
)


def _hue_shift(image: np.ndarray, shift: int) -> np.ndarray:
    hsv = cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = ((h.astype(int) + shift) % 180).astype("uint8")
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


register(
    Operation(
        key="equalize",
        label="Histogram equalise / CLAHE",
        category="Color",
        summary="Redistribute intensities to improve contrast. CLAHE does it in tiles.",
        docs="equalizeHist / createCLAHE",
        params=(
            Param("mode", "Method", "choice", "clahe", choices=("global", "clahe")),
            Param("clip", "CLAHE clip limit", "float", 2.0, 1.0, 8.0, 0.5),
            Param("tiles", "CLAHE tile grid", "int", 8, 2, 16, 1),
        ),
        run=lambda image, p: _equalize(image, p),
        code=lambda p: (
            [
                "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
                "gray = cv2.equalizeHist(gray)",
                "img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)",
            ]
            if p["mode"] == "global"
            else [
                "lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)",
                "l, a, b = cv2.split(lab)",
                f"clahe = cv2.createCLAHE(clipLimit={p['clip']}, tileGridSize=({p['tiles']}, {p['tiles']}))",
                "l = clahe.apply(l)",
                "img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)",
            ]
        ),
    )
)


def _equalize(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    if p["mode"] == "global":
        return to_bgr(cv2.equalizeHist(to_gray(image)))
    lab = cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=p["clip"], tileGridSize=(p["tiles"], p["tiles"]))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------
# Blur
# --------------------------------------------------------------------------

register(
    Operation(
        key="gaussian_blur",
        label="Gaussian blur",
        category="Blur",
        summary="Weighted average with a Gaussian kernel: the default smoother.",
        docs="GaussianBlur",
        params=(
            Param("ksize", "Kernel size", "int", 9, 1, 61, 2, odd_only=True,
                  help="Must be odd. Bigger blurs more."),
            Param("sigma", "Sigma X", "float", 0.0, 0.0, 20.0, 0.5,
                  help="0 lets OpenCV derive it from the kernel size."),
        ),
        run=lambda image, p: cv2.GaussianBlur(image, (p["ksize"], p["ksize"]), p["sigma"]),
        code=lambda p: [
            f"img = cv2.GaussianBlur(img, ({p['ksize']}, {p['ksize']}), {p['sigma']})"
        ],
    )
)

register(
    Operation(
        key="median_blur",
        label="Median blur",
        category="Blur",
        summary="Replaces each pixel with the median of its neighbours: kills salt-and-pepper noise.",
        docs="medianBlur",
        params=(Param("ksize", "Kernel size", "int", 7, 1, 31, 2, odd_only=True),),
        run=lambda image, p: cv2.medianBlur(image, p["ksize"]),
        code=lambda p: [f"img = cv2.medianBlur(img, {p['ksize']})"],
    )
)

register(
    Operation(
        key="bilateral",
        label="Bilateral filter",
        category="Blur",
        summary="Smooths flat areas but keeps edges sharp, by weighting on colour distance too.",
        docs="bilateralFilter",
        params=(
            Param("diameter", "Diameter", "int", 9, 1, 25, 2),
            Param("sigma_color", "Sigma colour", "int", 75, 1, 200, 1),
            Param("sigma_space", "Sigma space", "int", 75, 1, 200, 1),
        ),
        run=lambda image, p: cv2.bilateralFilter(
            to_bgr(image), p["diameter"], p["sigma_color"], p["sigma_space"]
        ),
        code=lambda p: [
            f"img = cv2.bilateralFilter(img, {p['diameter']}, {p['sigma_color']}, {p['sigma_space']})"
        ],
    )
)

register(
    Operation(
        key="sharpen",
        label="Sharpen (custom kernel)",
        category="Blur",
        summary="Convolution with a hand-written kernel, via filter2D.",
        docs="filter2D",
        params=(Param("amount", "Amount", "float", 1.0, 0.2, 3.0, 0.1),),
        run=lambda image, p: _sharpen(image, p["amount"]),
        code=lambda p: [
            "kernel = np.array([[ 0, -1,  0],",
            f"                   [-1, {4 + p['amount']:.2f}, -1],",
            "                   [ 0, -1,  0]], dtype=np.float32)",
            "kernel /= kernel.sum()",
            "img = cv2.filter2D(img, -1, kernel)",
        ],
    )
)


def _sharpen(image: np.ndarray, amount: float) -> np.ndarray:
    kernel = np.array(
        [[0, -1, 0], [-1, 4 + amount, -1], [0, -1, 0]], dtype=np.float32
    )
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


# --------------------------------------------------------------------------
# Threshold
# --------------------------------------------------------------------------

register(
    Operation(
        key="threshold",
        label="Threshold",
        category="Threshold",
        summary="Split pixels into black and white. Otsu picks the level for you.",
        docs="threshold",
        needs_gray=True,
        params=(
            Param("mode", "Method", "choice", "binary", choices=("binary", "binary_inv", "otsu", "trunc", "tozero")),
            Param("thresh", "Threshold", "int", 127, 0, 255, 1),
            Param("maxval", "Max value", "int", 255, 0, 255, 1),
        ),
        run=lambda image, p: _threshold(image, p),
        code=lambda p: [
            "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
            (
                f"_, mask = cv2.threshold(gray, 0, {p['maxval']}, cv2.THRESH_BINARY + cv2.THRESH_OTSU)"
                if p["mode"] == "otsu"
                else f"_, mask = cv2.threshold(gray, {p['thresh']}, {p['maxval']}, cv2.{_THRESH_FLAGS[p['mode']]})"
            ),
            "img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)",
        ],
    )
)

_THRESH_FLAGS = {
    "binary": "THRESH_BINARY",
    "binary_inv": "THRESH_BINARY_INV",
    "trunc": "THRESH_TRUNC",
    "tozero": "THRESH_TOZERO",
}


def _threshold(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    gray = to_gray(image)
    if p["mode"] == "otsu":
        _, mask = cv2.threshold(gray, 0, p["maxval"], cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        flag = getattr(cv2, _THRESH_FLAGS[p["mode"]])
        _, mask = cv2.threshold(gray, p["thresh"], p["maxval"], flag)
    return to_bgr(mask)


register(
    Operation(
        key="adaptive_threshold",
        label="Adaptive threshold",
        category="Threshold",
        summary="Chooses a threshold per neighbourhood, so uneven lighting stops mattering.",
        docs="adaptiveThreshold",
        needs_gray=True,
        params=(
            Param("method", "Method", "choice", "gaussian", choices=("mean", "gaussian")),
            Param("block", "Block size", "int", 15, 3, 61, 2, odd_only=True),
            Param("c", "Constant C", "int", 3, -20, 20, 1),
        ),
        run=lambda image, p: _adaptive(image, p),
        code=lambda p: [
            "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
            "mask = cv2.adaptiveThreshold(",
            f"    gray, 255, cv2.ADAPTIVE_THRESH_{'GAUSSIAN' if p['method'] == 'gaussian' else 'MEAN'}_C,",
            f"    cv2.THRESH_BINARY, {p['block']}, {p['c']},",
            ")",
            "img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)",
        ],
    )
)


def _adaptive(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C if p["method"] == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    mask = cv2.adaptiveThreshold(
        to_gray(image), 255, method, cv2.THRESH_BINARY, p["block"], p["c"]
    )
    return to_bgr(mask)


register(
    Operation(
        key="in_range",
        label="Colour mask (inRange)",
        category="Threshold",
        summary="Keep only pixels inside an HSV range: the basis of colour tracking.",
        docs="inRange",
        params=(
            Param("hue_low", "Hue low", "int", 35, 0, 179, 1),
            Param("hue_high", "Hue high", "int", 85, 0, 179, 1),
            Param("sat_low", "Saturation low", "int", 60, 0, 255, 1),
            Param("val_low", "Value low", "int", 60, 0, 255, 1),
            Param("show", "Output", "choice", "mask", choices=("mask", "masked")),
        ),
        run=lambda image, p: _in_range(image, p),
        code=lambda p: [
            "hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)",
            f"lower = np.array([{p['hue_low']}, {p['sat_low']}, {p['val_low']}])",
            f"upper = np.array([{p['hue_high']}, 255, 255])",
            "mask = cv2.inRange(hsv, lower, upper)",
            (
                "img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)"
                if p["show"] == "mask"
                else "img = cv2.bitwise_and(img, img, mask=mask)"
            ),
        ],
    )
)


def _in_range(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    bgr = to_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([p["hue_low"], p["sat_low"], p["val_low"]])
    upper = np.array([p["hue_high"], 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    if p["show"] == "mask":
        return to_bgr(mask)
    return cv2.bitwise_and(bgr, bgr, mask=mask)


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------

register(
    Operation(
        key="canny",
        label="Canny edges",
        category="Edges",
        summary="Gradient edges with hysteresis between two thresholds.",
        docs="Canny",
        needs_gray=True,
        params=(
            Param("low", "Lower threshold", "int", 80, 0, 500, 1),
            Param("high", "Upper threshold", "int", 180, 0, 500, 1),
            Param("aperture", "Sobel aperture", "int", 3, 3, 7, 2, odd_only=True),
        ),
        run=lambda image, p: to_bgr(
            cv2.Canny(to_gray(image), p["low"], p["high"], apertureSize=p["aperture"])
        ),
        code=lambda p: [
            "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
            f"edges = cv2.Canny(gray, {p['low']}, {p['high']}, apertureSize={p['aperture']})",
            "img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)",
        ],
    )
)

register(
    Operation(
        key="sobel",
        label="Sobel gradient",
        category="Edges",
        summary="First derivative in x, y, or both: shows where intensity changes.",
        docs="Sobel",
        needs_gray=True,
        params=(
            Param("direction", "Direction", "choice", "both", choices=("x", "y", "both")),
            Param("ksize", "Kernel size", "int", 3, 1, 7, 2, odd_only=True),
        ),
        run=lambda image, p: _sobel(image, p),
        code=lambda p: [
            "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
            *(
                [f"grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize={p['ksize']})"]
                if p["direction"] == "x"
                else [f"grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize={p['ksize']})"]
                if p["direction"] == "y"
                else [
                    f"gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize={p['ksize']})",
                    f"gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize={p['ksize']})",
                    "grad = cv2.magnitude(gx, gy)",
                ]
            ),
            "grad = cv2.convertScaleAbs(grad)",
            "img = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)",
        ],
    )
)


def _sobel(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    gray = to_gray(image)
    ksize = p["ksize"]
    if p["direction"] == "x":
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    elif p["direction"] == "y":
        grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    else:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        grad = cv2.magnitude(gx, gy)
    return to_bgr(cv2.convertScaleAbs(grad))


register(
    Operation(
        key="laplacian",
        label="Laplacian",
        category="Edges",
        summary="Second derivative: responds to intensity peaks and ridges.",
        docs="Laplacian",
        needs_gray=True,
        params=(Param("ksize", "Kernel size", "int", 3, 1, 31, 2, odd_only=True),),
        run=lambda image, p: to_bgr(
            cv2.convertScaleAbs(cv2.Laplacian(to_gray(image), cv2.CV_64F, ksize=p["ksize"]))
        ),
        code=lambda p: [
            "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
            f"lap = cv2.Laplacian(gray, cv2.CV_64F, ksize={p['ksize']})",
            "img = cv2.cvtColor(cv2.convertScaleAbs(lap), cv2.COLOR_GRAY2BGR)",
        ],
    )
)


# --------------------------------------------------------------------------
# Morphology
# --------------------------------------------------------------------------

register(
    Operation(
        key="morphology",
        label="Morphology",
        category="Morphology",
        summary="Erode, dilate, open or close using a structuring element.",
        docs="morphologyEx / getStructuringElement",
        params=(
            Param("op", "Operation", "choice", "open",
                  choices=("erode", "dilate", "open", "close", "gradient", "tophat", "blackhat")),
            Param("ksize", "Kernel size", "int", 5, 1, 31, 2, odd_only=True),
            Param("shape", "Kernel shape", "choice", "ellipse", choices=("rect", "ellipse", "cross")),
            Param("iterations", "Iterations", "int", 1, 1, 10, 1),
        ),
        run=lambda image, p: _morphology(image, p),
        code=lambda p: [
            f"kernel = cv2.getStructuringElement(cv2.MORPH_{p['shape'].upper()}, ({p['ksize']}, {p['ksize']}))",
            (
                f"img = cv2.erode(img, kernel, iterations={p['iterations']})"
                if p["op"] == "erode"
                else f"img = cv2.dilate(img, kernel, iterations={p['iterations']})"
                if p["op"] == "dilate"
                else f"img = cv2.morphologyEx(img, cv2.MORPH_{p['op'].upper()}, kernel, iterations={p['iterations']})"
            ),
        ],
    )
)

_MORPH_OPS = {
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "tophat": cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}


def _morphology(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    kernel = _kernel(p["ksize"], p["shape"])
    iterations = p["iterations"]
    if p["op"] == "erode":
        return cv2.erode(image, kernel, iterations=iterations)
    if p["op"] == "dilate":
        return cv2.dilate(image, kernel, iterations=iterations)
    return cv2.morphologyEx(image, _MORPH_OPS[p["op"]], kernel, iterations=iterations)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

register(
    Operation(
        key="resize",
        label="Resize",
        category="Geometry",
        summary="Scale the image; interpolation choice matters when enlarging.",
        docs="resize",
        params=(
            Param("scale", "Scale", "float", 0.5, 0.1, 2.0, 0.05),
            Param("interpolation", "Interpolation", "choice", "area",
                  choices=("nearest", "linear", "cubic", "area")),
        ),
        run=lambda image, p: cv2.resize(
            image, None, fx=p["scale"], fy=p["scale"], interpolation=_INTERP[p["interpolation"]]
        ),
        code=lambda p: [
            f"img = cv2.resize(img, None, fx={p['scale']}, fy={p['scale']},",
            f"                 interpolation=cv2.INTER_{p['interpolation'].upper()})",
        ],
    )
)

_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
}

register(
    Operation(
        key="rotate",
        label="Rotate",
        category="Geometry",
        summary="Affine rotation about the centre, via a 2x3 transform matrix.",
        docs="getRotationMatrix2D + warpAffine",
        params=(
            Param("angle", "Angle", "float", 15.0, -180.0, 180.0, 1.0),
            Param("scale", "Scale", "float", 1.0, 0.2, 2.0, 0.05),
        ),
        run=lambda image, p: _rotate(image, p),
        code=lambda p: [
            "h, w = img.shape[:2]",
            f"matrix = cv2.getRotationMatrix2D((w / 2, h / 2), {p['angle']}, {p['scale']})",
            "img = cv2.warpAffine(img, matrix, (w, h))",
        ],
    )
)


def _rotate(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), p["angle"], p["scale"])
    return cv2.warpAffine(image, matrix, (width, height))


register(
    Operation(
        key="flip",
        label="Flip",
        category="Geometry",
        summary="Mirror horizontally, vertically, or both.",
        docs="flip",
        params=(Param("axis", "Axis", "choice", "horizontal",
                      choices=("horizontal", "vertical", "both")),),
        run=lambda image, p: cv2.flip(image, _FLIP[p["axis"]]),
        code=lambda p: [f"img = cv2.flip(img, {_FLIP[p['axis']]})  # {p['axis']}"],
    )
)

_FLIP = {"horizontal": 1, "vertical": 0, "both": -1}

register(
    Operation(
        key="crop",
        label="Crop (ROI)",
        category="Geometry",
        summary="Slice a region of interest straight out of the numpy array.",
        docs="numpy slicing",
        params=(
            Param("x", "Left", "float", 0.15, 0.0, 0.9, 0.01),
            Param("y", "Top", "float", 0.15, 0.0, 0.9, 0.01),
            Param("w", "Width", "float", 0.7, 0.1, 1.0, 0.01),
            Param("h", "Height", "float", 0.7, 0.1, 1.0, 0.01),
        ),
        run=lambda image, p: _crop(image, p),
        code=lambda p: [
            "h, w = img.shape[:2]",
            f"x1, y1 = int({p['x']} * w), int({p['y']} * h)",
            f"x2, y2 = x1 + int({p['w']} * w), y1 + int({p['h']} * h)",
            "img = img[y1:y2, x1:x2]  # a view, not a copy",
        ],
    )
)


def _crop(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1 = int(p["x"] * width), int(p["y"] * height)
    x2 = min(width, x1 + max(1, int(p["w"] * width)))
    y2 = min(height, y1 + max(1, int(p["h"] * height)))
    cropped = image[y1:y2, x1:x2]
    return cropped if cropped.size else image


# --------------------------------------------------------------------------
# Features: contours, corners, lines, blobs
# --------------------------------------------------------------------------

register(
    Operation(
        key="contours",
        label="Find + draw contours",
        category="Features",
        summary="Trace object outlines, then optionally box them or mark their centroids.",
        docs="findContours / drawContours / boundingRect / moments",
        needs_gray=True,
        params=(
            Param("thresh", "Threshold", "int", 127, 0, 255, 1),
            Param("mode", "Retrieval", "choice", "external", choices=("external", "list", "tree")),
            Param("min_area", "Min area (px)", "int", 200, 0, 20000, 50,
                  help="Drops specks so the drawing stays readable."),
            Param("thickness", "Outline thickness", "int", 2, 1, 8, 1),
            Param("boxes", "Bounding boxes", "bool", True),
            Param("centroids", "Centroids", "bool", True),
            Param("labels", "Area labels", "bool", False),
        ),
        run=lambda image, p: _contours(image, p),
        code=lambda p: [
            "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
            f"_, mask = cv2.threshold(gray, {p['thresh']}, 255, cv2.THRESH_BINARY)",
            f"contours, _ = cv2.findContours(mask, cv2.RETR_{_RETRIEVAL[p['mode']]}, cv2.CHAIN_APPROX_SIMPLE)",
            f"contours = [c for c in contours if cv2.contourArea(c) >= {p['min_area']}]",
            f"cv2.drawContours(img, contours, -1, (0, 255, 0), {p['thickness']})",
            *(
                [
                    "for c in contours:",
                    "    x, y, w, h = cv2.boundingRect(c)",
                    "    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 0), 2)",
                ]
                if p["boxes"]
                else []
            ),
            *(
                [
                    "for c in contours:",
                    "    m = cv2.moments(c)",
                    "    if m['m00']:",
                    "        cx, cy = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])",
                    "        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)",
                ]
                if p["centroids"]
                else []
            ),
            *(
                [
                    "for c in contours:",
                    "    x, y, w, h = cv2.boundingRect(c)",
                    "    cv2.putText(img, f'{int(cv2.contourArea(c))}px', (x, y - 8),",
                    "                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)",
                ]
                if p["labels"]
                else []
            ),
        ],
    )
)

_RETRIEVAL = {"external": "EXTERNAL", "list": "LIST", "tree": "TREE"}
_RETRIEVAL_FLAGS = {
    "external": cv2.RETR_EXTERNAL,
    "list": cv2.RETR_LIST,
    "tree": cv2.RETR_TREE,
}


def _contours(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    output = to_bgr(image).copy()
    _, mask = cv2.threshold(to_gray(image), p["thresh"], 255, cv2.THRESH_BINARY)
    found, _ = cv2.findContours(mask, _RETRIEVAL_FLAGS[p["mode"]], cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in found if cv2.contourArea(c) >= p["min_area"]]
    cv2.drawContours(output, kept, -1, (0, 255, 0), p["thickness"])

    for contour in kept:
        if p["boxes"] or p["labels"]:
            x, y, w, h = cv2.boundingRect(contour)
            if p["boxes"]:
                cv2.rectangle(output, (x, y), (x + w, y + h), (255, 200, 0), 2)
            if p["labels"]:
                cv2.putText(
                    output,
                    f"{int(cv2.contourArea(contour))}px",
                    (x, max(14, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        if p["centroids"]:
            moments = cv2.moments(contour)
            if moments["m00"]:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
    return output


register(
    Operation(
        key="hough_lines",
        label="Hough lines",
        category="Features",
        summary="Find straight lines in an edge map and draw them.",
        docs="HoughLinesP",
        needs_gray=True,
        params=(
            Param("canny_low", "Canny low", "int", 80, 0, 400, 1),
            Param("canny_high", "Canny high", "int", 180, 0, 400, 1),
            Param("threshold", "Votes", "int", 80, 10, 400, 5),
            Param("min_length", "Min length", "int", 60, 5, 400, 5),
            Param("max_gap", "Max gap", "int", 10, 0, 100, 1),
        ),
        run=lambda image, p: _hough(image, p),
        code=lambda p: [
            "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
            f"edges = cv2.Canny(gray, {p['canny_low']}, {p['canny_high']})",
            "lines = cv2.HoughLinesP(",
            f"    edges, rho=1, theta=np.pi / 180, threshold={p['threshold']},",
            f"    minLineLength={p['min_length']}, maxLineGap={p['max_gap']},",
            ")",
            "for x1, y1, x2, y2 in (lines or np.empty((0, 1, 4), int))[:, 0]:",
            "    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)",
        ],
    )
)


def _hough(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    output = to_bgr(image).copy()
    edges = cv2.Canny(to_gray(image), p["canny_low"], p["canny_high"])
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=p["threshold"],
        minLineLength=p["min_length"],
        maxLineGap=p["max_gap"],
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
    return output


register(
    Operation(
        key="corners",
        label="Corner points",
        category="Features",
        summary="Shi-Tomasi corners, drawn as points: the classic feature detector.",
        docs="goodFeaturesToTrack",
        needs_gray=True,
        params=(
            Param("max_corners", "Max corners", "int", 80, 1, 500, 1),
            Param("quality", "Quality level", "float", 0.01, 0.001, 0.2, 0.005),
            Param("min_distance", "Min distance", "int", 10, 1, 60, 1),
            Param("radius", "Marker radius", "int", 4, 1, 14, 1),
        ),
        run=lambda image, p: _corners(image, p),
        code=lambda p: [
            "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
            "corners = cv2.goodFeaturesToTrack(",
            f"    gray, maxCorners={p['max_corners']}, qualityLevel={p['quality']},",
            f"    minDistance={p['min_distance']},",
            ")",
            "for x, y in np.int32(corners or []).reshape(-1, 2):",
            f"    cv2.circle(img, (x, y), {p['radius']}, (0, 0, 255), -1, cv2.LINE_AA)",
        ],
    )
)


def _corners(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    output = to_bgr(image).copy()
    corners = cv2.goodFeaturesToTrack(
        to_gray(image),
        maxCorners=p["max_corners"],
        qualityLevel=p["quality"],
        minDistance=p["min_distance"],
    )
    if corners is not None:
        for x, y in np.int32(corners).reshape(-1, 2):
            cv2.circle(output, (int(x), int(y)), p["radius"], (0, 0, 255), -1, cv2.LINE_AA)
    return output


# --------------------------------------------------------------------------
# Drawing: shapes, points, text
# --------------------------------------------------------------------------
# Coordinates are fractions of the image size so a shape stays put if the image
# is resized earlier in the pipeline; the generated code shows the arithmetic.

register(
    Operation(
        key="draw_rectangle",
        label="Rectangle",
        category="Drawing",
        summary="cv2.rectangle from two corners. Negative thickness fills it.",
        docs="rectangle",
        params=(
            Param("x1", "Left", "float", 0.18, 0.0, 1.0, 0.01),
            Param("y1", "Top", "float", 0.22, 0.0, 1.0, 0.01),
            Param("x2", "Right", "float", 0.62, 0.0, 1.0, 0.01),
            Param("y2", "Bottom", "float", 0.72, 0.0, 1.0, 0.01),
            Param("thickness", "Thickness", "int", 3, -1, 12, 1, help="-1 fills the shape."),
            Param("color", "Colour (BGR)", "color", (120, 255, 140)),
        ),
        run=lambda image, p: _draw_rectangle(image, p),
        code=lambda p: [
            "h, w = img.shape[:2]",
            f"pt1 = (int({p['x1']} * w), int({p['y1']} * h))",
            f"pt2 = (int({p['x2']} * w), int({p['y2']} * h))",
            f"cv2.rectangle(img, pt1, pt2, {tuple(p['color'])}, {p['thickness']})",
        ],
    )
)


def _draw_rectangle(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    output = to_bgr(image).copy()
    cv2.rectangle(
        output,
        _point(output, p["x1"], p["y1"]),
        _point(output, p["x2"], p["y2"]),
        tuple(p["color"]),
        p["thickness"],
        cv2.LINE_AA,
    )
    return output


register(
    Operation(
        key="draw_circle",
        label="Circle",
        category="Drawing",
        summary="cv2.circle from a centre and a radius in pixels.",
        docs="circle",
        params=(
            Param("cx", "Centre x", "float", 0.5, 0.0, 1.0, 0.01),
            Param("cy", "Centre y", "float", 0.5, 0.0, 1.0, 0.01),
            Param("radius", "Radius (px)", "int", 70, 2, 400, 2),
            Param("thickness", "Thickness", "int", 3, -1, 12, 1, help="-1 fills the shape."),
            Param("color", "Colour (BGR)", "color", (255, 231, 76)),
        ),
        run=lambda image, p: _draw_circle(image, p),
        code=lambda p: [
            "h, w = img.shape[:2]",
            f"centre = (int({p['cx']} * w), int({p['cy']} * h))",
            f"cv2.circle(img, centre, {p['radius']}, {tuple(p['color'])}, {p['thickness']}, cv2.LINE_AA)",
        ],
    )
)


def _draw_circle(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    output = to_bgr(image).copy()
    cv2.circle(
        output,
        _point(output, p["cx"], p["cy"]),
        p["radius"],
        tuple(p["color"]),
        p["thickness"],
        cv2.LINE_AA,
    )
    return output


register(
    Operation(
        key="draw_line",
        label="Line / arrow",
        category="Drawing",
        summary="cv2.line, or cv2.arrowedLine when you want a direction.",
        docs="line / arrowedLine",
        params=(
            Param("x1", "From x", "float", 0.15, 0.0, 1.0, 0.01),
            Param("y1", "From y", "float", 0.8, 0.0, 1.0, 0.01),
            Param("x2", "To x", "float", 0.8, 0.0, 1.0, 0.01),
            Param("y2", "To y", "float", 0.25, 0.0, 1.0, 0.01),
            Param("thickness", "Thickness", "int", 3, 1, 12, 1),
            Param("arrow", "Arrow head", "bool", True),
            Param("color", "Colour (BGR)", "color", (200, 60, 255)),
        ),
        run=lambda image, p: _draw_line(image, p),
        code=lambda p: [
            "h, w = img.shape[:2]",
            f"pt1 = (int({p['x1']} * w), int({p['y1']} * h))",
            f"pt2 = (int({p['x2']} * w), int({p['y2']} * h))",
            (
                f"cv2.arrowedLine(img, pt1, pt2, {tuple(p['color'])}, {p['thickness']}, tipLength=0.05)"
                if p["arrow"]
                else f"cv2.line(img, pt1, pt2, {tuple(p['color'])}, {p['thickness']}, cv2.LINE_AA)"
            ),
        ],
    )
)


def _draw_line(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    output = to_bgr(image).copy()
    pt1 = _point(output, p["x1"], p["y1"])
    pt2 = _point(output, p["x2"], p["y2"])
    if p["arrow"]:
        cv2.arrowedLine(output, pt1, pt2, tuple(p["color"]), p["thickness"], tipLength=0.05)
    else:
        cv2.line(output, pt1, pt2, tuple(p["color"]), p["thickness"], cv2.LINE_AA)
    return output


register(
    Operation(
        key="draw_polygon",
        label="Polygon / points",
        category="Drawing",
        summary="Build an array of points, then polylines or fillPoly it.",
        docs="polylines / fillPoly",
        params=(
            Param("sides", "Sides", "int", 5, 3, 12, 1),
            Param("radius", "Radius (px)", "int", 90, 10, 400, 5),
            Param("rotation", "Rotation", "float", 0.0, 0.0, 360.0, 5.0),
            Param("fill", "Fill", "bool", False),
            Param("markers", "Mark vertices", "bool", True),
            Param("color", "Colour (BGR)", "color", (60, 190, 255)),
        ),
        run=lambda image, p: _draw_polygon(image, p),
        code=lambda p: [
            "h, w = img.shape[:2]",
            "cx, cy = w // 2, h // 2",
            f"angles = np.linspace(0, 2 * np.pi, {p['sides']}, endpoint=False) + np.radians({p['rotation']})",
            f"pts = np.stack([cx + {p['radius']} * np.cos(angles), cy + {p['radius']} * np.sin(angles)], axis=1)",
            "pts = pts.astype(np.int32).reshape(-1, 1, 2)",
            (
                f"cv2.fillPoly(img, [pts], {tuple(p['color'])})"
                if p["fill"]
                else f"cv2.polylines(img, [pts], isClosed=True, color={tuple(p['color'])}, thickness=3)"
            ),
            *(
                [
                    "for x, y in pts.reshape(-1, 2):",
                    "    cv2.drawMarker(img, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)",
                ]
                if p["markers"]
                else []
            ),
        ],
    )
)


def _draw_polygon(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    output = to_bgr(image).copy()
    height, width = output.shape[:2]
    cx, cy = width // 2, height // 2
    angles = np.linspace(0, 2 * np.pi, p["sides"], endpoint=False) + np.radians(p["rotation"])
    points = np.stack(
        [cx + p["radius"] * np.cos(angles), cy + p["radius"] * np.sin(angles)], axis=1
    ).astype(np.int32).reshape(-1, 1, 2)

    if p["fill"]:
        cv2.fillPoly(output, [points], tuple(p["color"]))
    else:
        cv2.polylines(output, [points], True, tuple(p["color"]), 3, cv2.LINE_AA)
    if p["markers"]:
        for x, y in points.reshape(-1, 2):
            cv2.drawMarker(output, (int(x), int(y)), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
    return output


register(
    Operation(
        key="draw_text",
        label="Text",
        category="Drawing",
        summary="cv2.putText with a Hershey font. getTextSize measures it first.",
        docs="putText / getTextSize",
        params=(
            Param("text", "Text", "choice", "OpenCV",
                  choices=("OpenCV", "HELLO", "ROI", "detected", "6-7")),
            Param("x", "x", "float", 0.1, 0.0, 1.0, 0.01),
            Param("y", "y", "float", 0.2, 0.0, 1.0, 0.01),
            Param("scale", "Font scale", "float", 1.4, 0.3, 5.0, 0.1),
            Param("thickness", "Thickness", "int", 3, 1, 10, 1),
            Param("backdrop", "Filled backdrop", "bool", True,
                  help="Uses getTextSize to size a box behind the text."),
            Param("color", "Colour (BGR)", "color", (255, 255, 255)),
        ),
        run=lambda image, p: _draw_text(image, p),
        code=lambda p: [
            "h, w = img.shape[:2]",
            f"origin = (int({p['x']} * w), int({p['y']} * h))",
            "font = cv2.FONT_HERSHEY_SIMPLEX",
            *(
                [
                    f"(tw, th), base = cv2.getTextSize('{p['text']}', font, {p['scale']}, {p['thickness']})",
                    "cv2.rectangle(img, (origin[0] - 6, origin[1] - th - 6),",
                    "              (origin[0] + tw + 6, origin[1] + base + 6), (0, 0, 0), -1)",
                ]
                if p["backdrop"]
                else []
            ),
            f"cv2.putText(img, '{p['text']}', origin, font, {p['scale']},",
            f"            {tuple(p['color'])}, {p['thickness']}, cv2.LINE_AA)",
        ],
    )
)


def _draw_text(image: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    output = to_bgr(image).copy()
    origin = _point(output, p["x"], p["y"])
    font = cv2.FONT_HERSHEY_SIMPLEX
    if p["backdrop"]:
        (tw, th), base = cv2.getTextSize(p["text"], font, p["scale"], p["thickness"])
        cv2.rectangle(
            output,
            (origin[0] - 6, origin[1] - th - 6),
            (origin[0] + tw + 6, origin[1] + base + 6),
            (0, 0, 0),
            -1,
        )
    cv2.putText(
        output, p["text"], origin, font, p["scale"], tuple(p["color"]), p["thickness"], cv2.LINE_AA
    )
    return output


# --------------------------------------------------------------------------
# Registry access
# --------------------------------------------------------------------------

OPERATIONS: tuple[Operation, ...] = tuple(_OPERATIONS)
BY_KEY: dict[str, Operation] = {operation.key: operation for operation in OPERATIONS}


def get(key: str) -> Operation | None:
    return BY_KEY.get(key)


def catalog_json() -> list[dict[str, Any]]:
    """The whole catalog, grouped for the UI."""
    return [
        {
            "category": category,
            "operations": [op.to_json() for op in OPERATIONS if op.category == category],
        }
        for category in CATEGORIES
    ]
