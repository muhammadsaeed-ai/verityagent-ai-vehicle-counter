"""Command-line entry point for the VerityAgent AI Vehicle Counter."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_FALLBACK_FPS,
    DEFAULT_LINE_Y,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SUMMARY_PATH,
    DEFAULT_TRACKER,
    AppConfig,
)
from video_processor import VideoProcessingError, VideoProcessor


def _confidence(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("confidence must be a number between 0 and 1") from exc
    if not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("confidence must be between 0 and 1 inclusive")
    return result


def _non_negative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="app.py",
        description=(
            "Detect and track cars, motorcycles, buses, and trucks in a video; "
            "count confirmed crossings of a virtual horizontal line."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to the input video file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Annotated output video path")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH, help="JSON summary output path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics YOLO weights path or model identifier")
    parser.add_argument("--confidence", type=_confidence, default=DEFAULT_CONFIDENCE, help="Minimum detection confidence [0, 1]")
    parser.add_argument(
        "--line-y",
        type=_non_negative_int,
        default=DEFAULT_LINE_Y,
        help="Horizontal counting-line Y coordinate in pixels; omitted means half the frame height",
    )
    parser.add_argument("--direction", choices=("both", "positive", "negative"), default="both", help="Count both directions, top-to-bottom (positive), or bottom-to-top (negative)")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, help="Ultralytics tracker YAML, e.g. bytetrack.yaml")
    parser.add_argument("--headless", action="store_true", help="Process without opening a display window")
    parser.add_argument("--show-confidence", action="store_true", help="Include detection confidence in vehicle labels")
    parser.add_argument("--fallback-fps", type=float, default=DEFAULT_FALLBACK_FPS, help="Output FPS used if source FPS metadata is invalid")
    parser.add_argument("--verbose", action="store_true", help="Enable diagnostic logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the application and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fallback_fps <= 0:
        parser.error("--fallback-fps must be greater than zero")
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = AppConfig(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        model=args.model,
        confidence=args.confidence,
        line_y=args.line_y,
        direction=args.direction,
        headless=args.headless,
        show_confidence=args.show_confidence,
        tracker=args.tracker,
        fallback_fps=args.fallback_fps,
    )
    try:
        VideoProcessor(config).process()
    except (VideoProcessingError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
