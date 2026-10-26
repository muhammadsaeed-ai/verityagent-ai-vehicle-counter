"""YOLO vehicle detection and result parsing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from config import VEHICLE_CLASSES


class DetectionError(RuntimeError):
    """Raised when YOLO cannot be initialized or run inference."""


@dataclass(frozen=True, slots=True)
class VehicleDetection:
    """One vehicle detection returned by YOLO.

    Coordinates are pixel coordinates in (x1, y1, x2, y2) order. track_id
    is None when a detector result has no associated tracker identity.
    """

    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None

    @property
    def anchor(self) -> tuple[float, float]:
        """Return the bottom-center point used for line-crossing analysis."""
        x1, _, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, y2


def _to_list(value: Any) -> list[Any]:
    """Convert Torch tensors, NumPy arrays, and ordinary sequences to lists."""
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class VehicleDetector:
    """Load a pretrained Ultralytics model and parse supported vehicle results."""

    def __init__(self, model_path: str = "yolo26n.pt", confidence: float = 0.35) -> None:
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0 inclusive.")
        if not str(model_path).strip():
            raise ValueError("Model identifier or path must not be empty.")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectionError(
                "Ultralytics is not installed. Install project dependencies with "
                "'python -m pip install -r requirements.txt'."
            ) from exc
        try:
            self.model_path = str(model_path)
            self.confidence = float(confidence)
            self.model = YOLO(self.model_path)
            self.class_ids = self._resolve_vehicle_classes(self.model.names)
            self.class_id_set = tuple(self.class_ids.values())
            self.class_by_id = {class_id: name for name, class_id in self.class_ids.items()}
        except DetectionError:
            raise
        except Exception as exc:
            raise DetectionError(f"Could not initialize YOLO model '{model_path}': {exc}") from exc

    @staticmethod
    def _resolve_vehicle_classes(names: Mapping[int, str] | list[str]) -> dict[str, int]:
        """Resolve class IDs from the loaded model's actual names, not fixed COCO IDs."""
        items = names.items() if isinstance(names, Mapping) else enumerate(names)
        normalised: dict[str, int] = {}
        for class_id, name in items:
            canonical = str(name).strip().lower()
            if canonical in VEHICLE_CLASSES:
                normalised[canonical] = int(class_id)
        missing = [name for name in VEHICLE_CLASSES if name not in normalised]
        if missing:
            available = ", ".join(str(name) for _, name in items) if isinstance(names, Mapping) else ", ".join(map(str, names))
            raise DetectionError(
                "The selected model does not expose all required vehicle classes "
                f"({', '.join(missing)}). Available model classes: {available or 'none'}. "
                "Use a model whose class names include car, motorcycle, bus, and truck."
            )
        return normalised

    def detect(self, frame: np.ndarray) -> list[VehicleDetection]:
        """Run ordinary YOLO inference on a frame (without persistent tracking)."""
        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                classes=list(self.class_id_set),
                verbose=False,
            )
            return self.parse_result(results[0] if results else None)
        except DetectionError:
            raise
        except Exception as exc:
            raise DetectionError(f"YOLO inference failed: {exc}") from exc

    def parse_result(self, result: Any) -> list[VehicleDetection]:
        """Extract supported classes, bounding boxes, confidence, and optional IDs."""
        if result is None or getattr(result, "boxes", None) is None:
            return []
        boxes = result.boxes
        xyxy = _to_list(getattr(boxes, "xyxy", None))
        classes = _to_list(getattr(boxes, "cls", None))
        confidences = _to_list(getattr(boxes, "conf", None))
        ids_raw = getattr(boxes, "id", None)
        ids = _to_list(ids_raw) if ids_raw is not None else []
        parsed: list[VehicleDetection] = []
        for index, box in enumerate(xyxy):
            if index >= len(classes) or index >= len(confidences) or len(box) < 4:
                continue
            class_id = int(classes[index])
            class_name = self.class_by_id.get(class_id)
            if class_name is None:
                continue
            coords = tuple(float(point) for point in box[:4])
            confidence = float(confidences[index])
            if not all(math.isfinite(point) for point in coords) or not math.isfinite(confidence):
                continue
            track_id: int | None = None
            if index < len(ids) and ids[index] is not None:
                try:
                    candidate = float(ids[index])
                    if math.isfinite(candidate):
                        track_id = int(candidate)
                except (TypeError, ValueError, OverflowError):
                    track_id = None
            parsed.append(
                VehicleDetection(
                    bbox=(coords[0], coords[1], coords[2], coords[3]),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    track_id=track_id,
                )
            )
        return parsed
