"""Unique track counting when vehicles cross a virtual horizontal line."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from config import COUNTING_HYSTERESIS_PX, VEHICLE_CLASSES
from vehicle_detector import VehicleDetection


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    """A confirmed line-crossing event."""

    track_id: int
    vehicle_class: str
    direction: str  # positive means increasing image Y (top-to-bottom)


@dataclass(slots=True)
class _TrackState:
    stable_side: int
    vehicle_class: str
    counted_directions: set[str] = field(default_factory=set)


class VehicleCounter:
    """Count each track at most once per direction using a hysteresis band.

    positive is motion toward increasing image Y (top-to-bottom), while negative is
    motion toward decreasing image Y (bottom-to-top). Points within hysteresis_px of
    the line are considered on the line and do not change the remembered side.
    """

    def __init__(
        self,
        line_y: int,
        direction: str = "both",
        hysteresis_px: int = COUNTING_HYSTERESIS_PX,
    ) -> None:
        if isinstance(line_y, bool) or not isinstance(line_y, int) or line_y < 0:
            raise ValueError("line_y must be a non-negative integer.")
        if direction not in {"both", "positive", "negative"}:
            raise ValueError("direction must be 'both', 'positive', or 'negative'.")
        if isinstance(hysteresis_px, bool) or not isinstance(hysteresis_px, int) or hysteresis_px < 0:
            raise ValueError("hysteresis_px must be a non-negative integer.")
        self.line_y = line_y
        self.direction_filter = direction
        self.hysteresis_px = hysteresis_px
        self.counts: dict[str, int] = {name: 0 for name in VEHICLE_CLASSES}
        self._states: dict[int, _TrackState] = {}

    @property
    def total(self) -> int:
        """Return the total number of qualifying crossings counted so far."""
        return sum(self.counts.values())

    def update(self, vehicles: Iterable[VehicleDetection]) -> list[CrossingEvent]:
        """Update stable sides for observed vehicles and return new crossing events."""
        events: list[CrossingEvent] = []
        for vehicle in vehicles:
            track_id = vehicle.track_id
            if track_id is None or isinstance(track_id, bool):
                continue
            class_name = vehicle.class_name.lower()
            if class_name not in self.counts:
                continue
            _, anchor_y = vehicle.anchor
            side = self._stable_side(anchor_y)
            # A one-frame observation close to the line has no stable side and is ignored.
            if side == 0:
                continue
            state = self._states.get(track_id)
            if state is None:
                self._states[track_id] = _TrackState(side, class_name)
                continue
            if side == state.stable_side:
                state.vehicle_class = class_name
                continue

            crossed_direction = "positive" if side > state.stable_side else "negative"
            state.stable_side = side
            state.vehicle_class = class_name
            direction_allowed = (
                self.direction_filter == "both" or self.direction_filter == crossed_direction
            )
            if direction_allowed and crossed_direction not in state.counted_directions:
                state.counted_directions.add(crossed_direction)
                self.counts[class_name] += 1
                events.append(CrossingEvent(int(track_id), class_name, crossed_direction))
        return events

    def reset(self) -> None:
        """Clear all counts and learned track sides."""
        for name in self.counts:
            self.counts[name] = 0
        self._states.clear()

    def _stable_side(self, y: float) -> int:
        if y < self.line_y - self.hysteresis_px:
            return -1
        if y > self.line_y + self.hysteresis_px:
            return 1
        return 0
