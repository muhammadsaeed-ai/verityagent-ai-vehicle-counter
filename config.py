"""Configuration and defaults for VerityAgent AI Vehicle Counter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_MODEL: Final[str] = "yolo26n.pt"
DEFAULT_CONFIDENCE: Final[float] = 0.35
DEFAULT_TRACKER: Final[str] = "bytetrack.yaml"
DEFAULT_OUTPUT_PATH: Final[Path] = Path("outputs/traffic_annotated.mp4")
DEFAULT_SUMMARY_PATH: Final[Path] = Path("outputs/traffic_summary.json")
DEFAULT_LINE_Y: Final[int | None] = None  # None means half the input frame height.
DEFAULT_FALLBACK_FPS: Final[float] = 25.0
COUNTING_HYSTERESIS_PX: Final[int] = 8

VEHICLE_CLASSES: Final[tuple[str, ...]] = ("car", "motorcycle", "bus", "truck")
# BGR colors for OpenCV drawing.
CLASS_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "car": (255, 190, 70),
    "motorcycle": (90, 220, 255),
    "bus": (100, 220, 120),
    "truck": (220, 130, 255),
}
OVERLAY_COLOR: Final[tuple[int, int, int]] = (245, 245, 245)
LINE_COLOR: Final[tuple[int, int, int]] = (0, 220, 255)
BOX_THICKNESS: Final[int] = 2
FONT_SCALE: Final[float] = 0.55
FONT_THICKNESS: Final[int] = 2


@dataclass(slots=True)
class AppConfig:
    """Runtime settings for one video-processing job."""

    input_path: Path
    output_path: Path = DEFAULT_OUTPUT_PATH
    summary_path: Path = DEFAULT_SUMMARY_PATH
    model: str = DEFAULT_MODEL
    confidence: float = DEFAULT_CONFIDENCE
    line_y: int | None = DEFAULT_LINE_Y
    direction: str = "both"
    headless: bool = False
    show_confidence: bool = False
    tracker: str = DEFAULT_TRACKER
    fallback_fps: float = DEFAULT_FALLBACK_FPS
    hysteresis_px: int = COUNTING_HYSTERESIS_PX

    def validate(self) -> None:
        """Validate values independent of the video frame dimensions."""
        if not str(self.input_path).strip():
            raise ValueError("Input video path must not be empty.")
        if not str(self.output_path).strip():
            raise ValueError("Output video path must not be empty.")
        if not str(self.summary_path).strip():
            raise ValueError("Summary JSON path must not be empty.")
        if not str(self.model).strip():
            raise ValueError("Model identifier must not be empty.")
        if not str(self.tracker).strip():
            raise ValueError("Tracker configuration must not be empty.")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0 inclusive.")
        if self.line_y is not None and (isinstance(self.line_y, bool) or self.line_y < 0):
            raise ValueError("Line Y must be a non-negative integer or None.")
        if self.line_y is not None and not isinstance(self.line_y, int):
            raise ValueError("Line Y must be a non-negative integer or None.")
        if self.direction not in {"both", "positive", "negative"}:
            raise ValueError("Direction must be 'both', 'positive', or 'negative'.")
        if self.fallback_fps <= 0:
            raise ValueError("Fallback FPS must be greater than zero.")
        if self.hysteresis_px < 0:
            raise ValueError("Counting hysteresis must be non-negative.")
