"""Tests for virtual-line crossing logic."""

import pytest

from vehicle_counter import VehicleCounter
from vehicle_detector import VehicleDetection


def vehicle(track_id, class_name="car", anchor_y=40):
    class_ids = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
    return VehicleDetection(
        bbox=(10.0, float(anchor_y - 20), 30.0, float(anchor_y)),
        class_id=class_ids[class_name],
        class_name=class_name,
        confidence=0.91,
        track_id=track_id,
    )


def test_vehicle_that_does_not_cross_is_not_counted():
    counter = VehicleCounter(line_y=50, hysteresis_px=5)
    counter.update([vehicle(1, anchor_y=20)])
    counter.update([vehicle(1, anchor_y=30)])
    counter.update([vehicle(1, anchor_y=44)])
    assert counter.total == 0


def test_vehicle_crossing_line_is_counted_once():
    counter = VehicleCounter(line_y=50, hysteresis_px=5)
    counter.update([vehicle(7, anchor_y=30)])
    events = counter.update([vehicle(7, anchor_y=70)])
    assert len(events) == 1
    assert events[0].direction == "positive"
    assert counter.counts["car"] == 1
    assert counter.total == 1


def test_same_track_is_not_counted_repeatedly_in_same_direction():
    counter = VehicleCounter(line_y=50, direction="positive", hysteresis_px=5)
    for y in (25, 75, 25, 72, 28, 80):
        counter.update([vehicle(9, anchor_y=y)])
    assert counter.counts["car"] == 1
    assert counter.total == 1


def test_different_vehicle_types_increment_separate_counts():
    counter = VehicleCounter(line_y=50, hysteresis_px=5)
    for track_id, kind in enumerate(("car", "motorcycle", "bus", "truck"), start=1):
        counter.update([vehicle(track_id, kind, anchor_y=20)])
        counter.update([vehicle(track_id, kind, anchor_y=80)])
    assert counter.counts == {"car": 1, "motorcycle": 1, "bus": 1, "truck": 1}
    assert counter.total == 4


def test_total_matches_sum_of_class_counts():
    counter = VehicleCounter(line_y=50, hysteresis_px=0)
    counter.update([vehicle(1, "car", 10), vehicle(2, "bus", 15)])
    counter.update([vehicle(1, "car", 90), vehicle(2, "bus", 95)])
    assert counter.total == sum(counter.counts.values()) == 2


def test_direction_filters_only_count_requested_direction():
    positive = VehicleCounter(line_y=50, direction="positive", hysteresis_px=2)
    positive.update([vehicle(1, anchor_y=20)])
    positive.update([vehicle(1, anchor_y=80)])
    positive.update([vehicle(2, anchor_y=80)])
    positive.update([vehicle(2, anchor_y=20)])
    assert positive.counts["car"] == 1

    negative = VehicleCounter(line_y=50, direction="negative", hysteresis_px=2)
    negative.update([vehicle(1, anchor_y=20)])
    negative.update([vehicle(1, anchor_y=80)])
    negative.update([vehicle(1, anchor_y=20)])
    assert negative.counts["car"] == 1


def test_invalid_line_configuration_is_rejected():
    with pytest.raises(ValueError, match="line_y"):
        VehicleCounter(line_y=-1)
    with pytest.raises(ValueError, match="direction"):
        VehicleCounter(line_y=10, direction="sideways")
    with pytest.raises(ValueError, match="hysteresis"):
        VehicleCounter(line_y=10, hysteresis_px=-1)


def test_reset_clears_counts_and_track_sides():
    counter = VehicleCounter(line_y=50, hysteresis_px=2)
    counter.update([vehicle(1, anchor_y=20)])
    counter.update([vehicle(1, anchor_y=80)])
    counter.reset()
    assert counter.total == 0
    assert all(count == 0 for count in counter.counts.values())
    counter.update([vehicle(1, anchor_y=20)])
    counter.update([vehicle(1, anchor_y=80)])
    assert counter.total == 1


def test_missing_track_id_does_not_crash_or_count():
    counter = VehicleCounter(line_y=50)
    counter.update([vehicle(None, anchor_y=20)])
    counter.update([vehicle(None, anchor_y=80)])
    assert counter.total == 0


def test_a_single_observation_on_the_line_is_not_a_crossing():
    counter = VehicleCounter(line_y=50, hysteresis_px=5)
    counter.update([vehicle(1, anchor_y=50)])
    assert counter.total == 0
