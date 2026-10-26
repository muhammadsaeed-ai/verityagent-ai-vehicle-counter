"""Video pipeline tests using mock capture, writer, and tracking components."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from config import AppConfig
from vehicle_detector import VehicleDetection
from video_processor import VideoProcessingError, VideoProcessor


class FakeCapture:
    def __init__(self, frames=None, opened=True, width=64, height=48, fps=24.0, frame_count=None):
        self.frames = list(frames if frames is not None else [np.zeros((height, width, 3), dtype=np.uint8)])
        self.opened = opened
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = len(self.frames) if frame_count is None else frame_count
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: self.frame_count,
        }.get(prop, 0)

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.frames = []

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class FakeTracker:
    def __init__(self, results=None):
        self.results = list(results if results is not None else [[]])
        self.calls = 0

    def update(self, frame):
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return []


def make_vehicle(track_id, y, class_name="car"):
    class_ids = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
    return VehicleDetection((10, y - 10, 20, y), class_ids[class_name], class_name, 0.88, track_id)


def patch_video_io(monkeypatch, capture, writer):
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: writer)
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *args: 0)


def test_invalid_input_path_is_reported_and_capture_released(tmp_path, monkeypatch):
    capture = FakeCapture(opened=False)
    writer = FakeWriter()
    patch_video_io(monkeypatch, capture, writer)
    config = AppConfig(tmp_path / "missing.mp4", tmp_path / "out.mp4", tmp_path / "summary.json", headless=True)
    with pytest.raises(VideoProcessingError, match="Could not open input video"):
        VideoProcessor(config, detector=object(), tracker=FakeTracker()).process()
    assert capture.released
    assert not writer.released


def test_invalid_video_metadata_is_handled(tmp_path, monkeypatch):
    capture = FakeCapture(width=0)
    writer = FakeWriter()
    patch_video_io(monkeypatch, capture, writer)
    config = AppConfig(tmp_path / "input.mp4", tmp_path / "out.mp4", tmp_path / "summary.json", headless=True)
    with pytest.raises(VideoProcessingError, match="Invalid video dimensions"):
        VideoProcessor(config, detector=object(), tracker=FakeTracker()).process()
    assert capture.released


def test_output_writer_initialization_failure_releases_capture_and_writer(tmp_path, monkeypatch):
    capture = FakeCapture()
    writer = FakeWriter(opened=False)
    patch_video_io(monkeypatch, capture, writer)
    config = AppConfig(tmp_path / "input.mp4", tmp_path / "out.mp4", tmp_path / "summary.json", headless=True)
    with pytest.raises(VideoProcessingError, match="Could not initialize output video writer"):
        VideoProcessor(config, detector=object(), tracker=FakeTracker()).process()
    assert capture.released and writer.released


def test_headless_mode_processes_frames_without_display_and_saves_summary(tmp_path, monkeypatch):
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
    capture = FakeCapture(frames=frames, fps=float("nan"), frame_count=3)
    writer = FakeWriter()
    tracker = FakeTracker([
        [make_vehicle(1, 15)],
        [make_vehicle(1, 40)],
        [make_vehicle(1, 44)],
    ])
    patch_video_io(monkeypatch, capture, writer)
    monkeypatch.setattr(cv2, "imshow", lambda *args: pytest.fail("imshow must not be used in headless mode"))
    config = AppConfig(
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "out.mp4",
        summary_path=tmp_path / "summary.json",
        line_y=24,
        headless=True,
    )
    summary = VideoProcessor(config, detector=object(), tracker=tracker).process()
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary == saved
    assert summary["processed_frames"] == 3
    assert summary["source_fps"] is None
    assert summary["output_fps"] == config.fallback_fps
    assert summary["vehicle_counts"]["car"] == 1
    assert summary["total_vehicles"] == sum(summary["vehicle_counts"].values())
    assert len(writer.frames) == 3
    assert capture.released and writer.released


def test_resource_cleanup_runs_after_tracking_exception(tmp_path, monkeypatch):
    capture = FakeCapture()
    writer = FakeWriter()
    patch_video_io(monkeypatch, capture, writer)

    class BrokenTracker:
        def update(self, frame):
            raise RuntimeError("tracking broke")

    config = AppConfig(tmp_path / "input.mp4", tmp_path / "out.mp4", tmp_path / "summary.json", headless=True)
    with pytest.raises(VideoProcessingError, match="tracking broke"):
        VideoProcessor(config, detector=object(), tracker=BrokenTracker()).process()
    assert capture.released and writer.released


def test_summary_serialization_writes_json_atomically(tmp_path):
    target = tmp_path / "nested" / "report.json"
    VideoProcessor.write_summary(target, {"processed_frames": 5, "vehicle_counts": {"car": 2}})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "processed_frames": 5,
        "vehicle_counts": {"car": 2},
    }
    assert not Path(str(target) + ".tmp").exists()


def test_line_outside_frame_is_rejected_and_capture_released(tmp_path, monkeypatch):
    capture = FakeCapture(height=48)
    writer = FakeWriter()
    patch_video_io(monkeypatch, capture, writer)
    config = AppConfig(tmp_path / "input.mp4", tmp_path / "out.mp4", tmp_path / "summary.json", line_y=99, headless=True)
    with pytest.raises(VideoProcessingError, match="outside the frame height"):
        VideoProcessor(config, detector=object(), tracker=FakeTracker()).process()
    assert capture.released


def test_output_and_summary_paths_must_be_distinct(tmp_path):
    shared = tmp_path / "same.mp4"
    config = AppConfig(tmp_path / "input.mp4", shared, shared, headless=True)
    processor = VideoProcessor(config, detector=object(), tracker=FakeTracker())
    with pytest.raises(VideoProcessingError, match="must be different"):
        processor.process()


def test_input_must_not_be_overwritten_by_summary(tmp_path):
    shared = tmp_path / "input.mp4"
    config = AppConfig(shared, tmp_path / "out.mp4", shared, headless=True)
    processor = VideoProcessor(config, detector=object(), tracker=FakeTracker())
    with pytest.raises(VideoProcessingError, match="Input video and summary JSON"):
        processor.process()
