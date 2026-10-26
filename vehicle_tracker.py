"""Persistent multi-object tracking using Ultralytics' official tracking API."""

from __future__ import annotations

from typing import Any

import numpy as np

from vehicle_detector import DetectionError, VehicleDetection, VehicleDetector


class TrackingError(RuntimeError):
    """Raised when the Ultralytics tracking API fails."""


class VehicleTracker:
    """Track vehicles across sequential frames using a configured Ultralytics tracker."""

    def __init__(self, detector: VehicleDetector, tracker_config: str = "bytetrack.yaml") -> None:
        if not str(tracker_config).strip():
            raise ValueError("Tracker configuration must not be empty.")
        self.detector = detector
        self.tracker_config = str(tracker_config)
        self._started = False

    def update(self, frame: np.ndarray) -> list[VehicleDetection]:
        """Run YOLO tracking on the next frame, preserving track state between frames."""
        try:
            results: Any = self.detector.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.detector.confidence,
                classes=list(self.detector.class_id_set),
                verbose=False,
            )
            self._started = True
            return self.detector.parse_result(results[0] if results else None)
        except (DetectionError, TrackingError):
            raise
        except Exception as exc:
            raise TrackingError(f"Ultralytics tracking failed: {exc}") from exc

    def reset(self) -> None:
        """Reset this wrapper's state marker.

        A fresh tracker instance is recommended when processing another video because
        Ultralytics stores the active tracker inside the model predictor. VideoProcessor
        creates one tracker for exactly one input stream.
        """
        self._started = False
