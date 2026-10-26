"""Tests for parsing and wrapping Ultralytics tracking results without model downloads."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vehicle_detector import DetectionError, VehicleDetector
from vehicle_tracker import TrackingError, VehicleTracker


class FakeBoxes:
    def __init__(self, xyxy=None, cls=None, conf=None, ids=None):
        self.xyxy = np.asarray(xyxy if xyxy is not None else [], dtype=float).reshape(-1, 4)
        self.cls = np.asarray(cls if cls is not None else [], dtype=float)
        self.conf = np.asarray(conf if conf is not None else [], dtype=float)
        self.id = None if ids is None else np.asarray(ids, dtype=float)


def fake_detector(result=None):
    detector = object.__new__(VehicleDetector)
    detector.confidence = 0.35
    detector.class_id_set = (2, 3, 5, 7)
    detector.class_by_id = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    detector.model = SimpleNamespace(track=Mock(return_value=[result]))
    return detector


def test_empty_tracking_result_is_handled():
    detector = fake_detector(SimpleNamespace(boxes=FakeBoxes()))
    tracker = VehicleTracker(detector)
    assert tracker.update(np.zeros((64, 64, 3), dtype=np.uint8)) == []


def test_missing_track_ids_are_handled():
    boxes = FakeBoxes([[1, 2, 20, 30]], [2], [0.8], ids=None)
    detector = fake_detector(SimpleNamespace(boxes=boxes))
    vehicles = VehicleTracker(detector).update(np.zeros((64, 64, 3), dtype=np.uint8))
    assert len(vehicles) == 1
    assert vehicles[0].track_id is None


def test_detection_results_are_parsed_correctly():
    boxes = FakeBoxes([[1, 2, 20, 30], [5, 6, 25, 40]], [2, 7], [0.81, 0.93], ids=[11, 12])
    detector = fake_detector(SimpleNamespace(boxes=boxes))
    vehicles = VehicleTracker(detector).update(np.zeros((64, 64, 3), dtype=np.uint8))
    assert [(v.class_name, v.track_id) for v in vehicles] == [("car", 11), ("truck", 12)]
    assert vehicles[0].bbox == (1.0, 2.0, 20.0, 30.0)
    assert vehicles[1].confidence == pytest.approx(0.93)
    detector.model.track.assert_called_once()
    assert detector.model.track.call_args.kwargs["persist"] is True
    assert detector.model.track.call_args.kwargs["tracker"] == "bytetrack.yaml"


def test_tracker_integration_errors_are_reported_clearly():
    detector = fake_detector()
    detector.model.track.side_effect = RuntimeError("backend unavailable")
    tracker = VehicleTracker(detector)
    with pytest.raises(TrackingError, match="backend unavailable"):
        tracker.update(np.zeros((64, 64, 3), dtype=np.uint8))


def test_detector_parser_ignores_non_vehicle_and_malformed_rows():
    detector = object.__new__(VehicleDetector)
    detector.class_by_id = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.asarray([[1, 2, 20, 30], [1, 2, 12, 16]], dtype=float),
            cls=np.asarray([2, 0], dtype=float),
            conf=np.asarray([0.7, 0.8], dtype=float),
            id=None,
        )
    )
    detections = detector.parse_result(result)
    assert len(detections) == 1
    assert detections[0].class_name == "car"
    assert detections[0].track_id is None


def test_bad_tracker_configuration_is_rejected():
    with pytest.raises(ValueError, match="Tracker configuration"):
        VehicleTracker(fake_detector(), tracker_config=" ")


def test_detection_error_type_is_importable_for_pipeline_wrapping():
    assert issubclass(DetectionError, RuntimeError)
