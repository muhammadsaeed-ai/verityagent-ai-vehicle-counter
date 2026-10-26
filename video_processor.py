"""OpenCV video pipeline for YOLO detection, tracking, counting, and export."""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import (
    BOX_THICKNESS,
    CLASS_COLORS,
    FONT_SCALE,
    FONT_THICKNESS,
    LINE_COLOR,
    OVERLAY_COLOR,
    AppConfig,
    VEHICLE_CLASSES,
)
from vehicle_counter import VehicleCounter
from vehicle_detector import DetectionError, VehicleDetection, VehicleDetector
from vehicle_tracker import TrackingError, VehicleTracker

LOGGER = logging.getLogger(__name__)


class VideoProcessingError(RuntimeError):
    """Raised for invalid video input or failures during video processing/export."""


class VideoProcessor:
    """Process one video into an annotated video and a machine-readable JSON summary."""

    def __init__(
        self,
        config: AppConfig,
        detector: VehicleDetector | None = None,
        tracker: VehicleTracker | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.detector = detector
        self.tracker = tracker

    def process(self) -> dict[str, Any]:
        """Process every readable frame and return the saved summary dictionary."""
        capture = None
        writer = None
        display_initialized = False
        started_at = time.monotonic()
        source_fps: float | None = None
        output_fps = self.config.fallback_fps
        processed_frames = 0
        output_path = Path(self.config.output_path)
        summary_path = Path(self.config.summary_path)
        input_path = Path(self.config.input_path)
        if input_path.resolve() == output_path.resolve():
            raise VideoProcessingError("Input and output video paths must be different.")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            capture = cv2.VideoCapture(str(input_path))
            if not capture.isOpened():
                raise VideoProcessingError(f"Could not open input video: {input_path}")

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
            metadata_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0:
                raise VideoProcessingError(
                    f"Invalid video dimensions reported by OpenCV: {width}x{height}."
                )
            if math.isfinite(reported_fps) and reported_fps > 0:
                source_fps = reported_fps
                output_fps = reported_fps
            else:
                LOGGER.warning(
                    "Source FPS is unavailable or invalid; using fallback output FPS %.2f.",
                    self.config.fallback_fps,
                )
            line_y = self.config.line_y if self.config.line_y is not None else height // 2
            if not 0 <= line_y < height:
                raise VideoProcessingError(
                    f"Counting line Y={line_y} is outside the frame height (0..{height - 1})."
                )

            if self.detector is None:
                self.detector = VehicleDetector(self.config.model, self.config.confidence)
            if self.tracker is None:
                self.tracker = VehicleTracker(self.detector, self.config.tracker)
            counter = VehicleCounter(
                line_y=line_y,
                direction=self.config.direction,
                hysteresis_px=self.config.hysteresis_px,
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
            if not writer.isOpened():
                raise VideoProcessingError(
                    f"Could not initialize output video writer for '{output_path}'. "
                    "Check the output extension, codec support, and directory permissions."
                )

            print(
                f"Processing {input_path} ({width}x{height}, "
                f"source FPS={source_fps if source_fps is not None else 'unavailable'}, "
                f"output FPS={output_fps:.2f})"
            )
            progress_step = max(1, metadata_frame_count // 20) if metadata_frame_count > 0 else 100
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    LOGGER.warning("OpenCV returned an empty frame; stopping at frame %d.", processed_frames + 1)
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    raise VideoProcessingError(
                        "A decoded frame does not match the dimensions reported by the video metadata."
                    )

                try:
                    vehicles = self.tracker.update(frame)
                except (DetectionError, TrackingError) as exc:
                    raise VideoProcessingError(str(exc)) from exc
                except Exception as exc:
                    raise VideoProcessingError(f"Tracking/detection step failed: {exc}") from exc
                counter.update(vehicles)
                annotated = self._annotate(frame, vehicles, counter, line_y)
                try:
                    writer.write(annotated)
                except Exception as exc:
                    raise VideoProcessingError(f"Failed while writing output frame: {exc}") from exc
                processed_frames += 1

                if not self.config.headless:
                    try:
                        cv2.imshow("VerityAgent AI Vehicle Counter", annotated)
                        display_initialized = True
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Display requested early stop (q); saving processed-frame summary.")
                            break
                    except cv2.error as exc:
                        raise VideoProcessingError(
                            "Could not open the display window. Re-run with --headless for "
                            f"display-free processing. Details: {exc}"
                        ) from exc

                if processed_frames == 1 or processed_frames % progress_step == 0:
                    if metadata_frame_count > 0:
                        percent = min(100.0, processed_frames * 100.0 / metadata_frame_count)
                        print(
                            f"\rProgress: {processed_frames}/{metadata_frame_count} frames "
                            f"({percent:.1f}%) | counted: {counter.total}",
                            end="",
                            flush=True,
                        )
                    else:
                        print(f"\rProcessed {processed_frames} frames | counted: {counter.total}", end="", flush=True)

            if processed_frames == 0:
                raise VideoProcessingError(f"No decodable video frames were read from '{input_path}'.")
            print()
            summary = {
                "input_video": input_path.name,
                "model": self.config.model,
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "processed_frames": processed_frames,
                "source_fps": source_fps,
                "output_fps": output_fps,
                "frame_width": width,
                "frame_height": height,
                "vehicle_counts": {name: int(counter.counts[name]) for name in VEHICLE_CLASSES},
                "total_vehicles": int(counter.total),
                "counting_line": {
                    "orientation": "horizontal",
                    "y": line_y,
                    "hysteresis_px": self.config.hysteresis_px,
                    "direction_filter": self.config.direction,
                    "positive_direction": "top-to-bottom (increasing image Y)",
                    "negative_direction": "bottom-to-top (decreasing image Y)",
                },
                "tracker": self.config.tracker,
                "confidence_threshold": self.config.confidence,
                "output_video": str(output_path),
                "headless": self.config.headless,
                "processing_seconds": round(time.monotonic() - started_at, 3),
            }
            self.write_summary(summary_path, summary)
            print(f"Finished: {processed_frames} frames processed, {counter.total} vehicle crossing(s) counted.")
            print(f"Annotated video: {output_path}")
            print(f"JSON summary:    {summary_path}")
            return summary
        except VideoProcessingError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise VideoProcessingError(str(exc)) from exc
        finally:
            # Cleanup operations are deliberately independent so one failure cannot skip another.
            if writer is not None:
                try:
                    writer.release()
                except Exception as exc:  # pragma: no cover - defensive for backend-specific cleanup.
                    LOGGER.warning("Could not release video writer cleanly: %s", exc)
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:  # pragma: no cover - defensive for backend-specific cleanup.
                    LOGGER.warning("Could not release video capture cleanly: %s", exc)
            if not self.config.headless and display_initialized:
                try:
                    cv2.destroyAllWindows()
                except cv2.error as exc:  # pragma: no cover - GUI backends differ by platform.
                    LOGGER.warning("Could not close OpenCV display windows cleanly: %s", exc)

    @staticmethod
    def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
        """Write a JSON summary atomically to avoid leaving a partial file on failure."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(target)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Could not remove temporary summary file '%s'.", temporary)
            raise VideoProcessingError(f"Could not save JSON summary to '{target}': {exc}") from exc

    def _annotate(
        self,
        frame: np.ndarray,
        vehicles: list[VehicleDetection],
        counter: VehicleCounter,
        line_y: int,
    ) -> np.ndarray:
        """Draw vehicle boxes, IDs, line, and current count overlay on a copy of frame."""
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        cv2.line(annotated, (0, line_y), (width - 1, line_y), LINE_COLOR, 2)
        for vehicle in vehicles:
            x1, y1, x2, y2 = (int(round(value)) for value in vehicle.bbox)
            x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
            y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            color = CLASS_COLORS.get(vehicle.class_name, OVERLAY_COLOR)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
            label = vehicle.class_name.title()
            if vehicle.track_id is not None:
                label += f" ID:{vehicle.track_id}"
            if self.config.show_confidence:
                label += f" {vehicle.confidence:.2f}"
            label_y = max(18, y1 - 7)
            cv2.putText(
                annotated,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                color,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        lines = [
            f"CARS {counter.counts['car']}   MOTO {counter.counts['motorcycle']}",
            f"BUSES {counter.counts['bus']}   TRUCKS {counter.counts['truck']}",
            f"TOTAL {counter.total}   DIR {self.config.direction.upper()}",
        ]
        x, y = 12, 25
        padding = 9
        line_height = 23
        panel_width = min(width - 1, 300)
        panel_height = len(lines) * line_height + padding * 2
        cv2.rectangle(annotated, (5, 5), (5 + panel_width, 5 + panel_height), (28, 32, 42), -1)
        for index, text in enumerate(lines):
            cv2.putText(
                annotated,
                text,
                (x, y + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                OVERLAY_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )
        return annotated
