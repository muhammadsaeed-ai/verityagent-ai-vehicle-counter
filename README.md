# AI Vehicle Counter — Python, OpenCV & YOLO

**VerityAgent Python AI Mini Projects — Project #06**

An AI-powered traffic-video analytics application. It uses a pretrained Ultralytics YOLO model to detect cars, motorcycles, buses, and trucks, tracks vehicle identities across frames with ByteTrack, counts crossings of a configurable virtual line, saves an annotated MP4, and exports a JSON traffic summary.

> Real model inference occurs when the app is run with the runtime dependencies, model weights, and an input video. No model weights, sample videos, or fabricated results are included in this repository.

## Features

- Pretrained YOLO detection with the official Ultralytics Python API.
- Persistent tracking through the Ultralytics model.track API with persistence enabled and ByteTrack.
- Class IDs resolved from the loaded model's actual class names.
- Bottom-center anchor and horizontal line-crossing logic with a hysteresis band.
- Direction filter: both directions, top-to-bottom (positive), or bottom-to-top (negative).
- Per-class and total traffic counts.
- Bounding boxes, labels, tracking IDs, optional confidence scores, and a live count overlay.
- Annotated MP4 output and structured JSON summary.
- Headless processing for servers and automated pipelines.
- Unit tests mock model/tracker and video I/O; no model download or traffic clip is required for tests.

## Architecture

~~~mermaid
flowchart TD
    A[Input video] --> B[OpenCV VideoCapture]
    B --> C[YOLO inference + persistent ByteTrack]
    C --> D[Supported vehicle classes]
    D --> E[Track anchor / line crossing]
    E --> F[Per-class and total counts]
    D --> G[Bounding boxes / labels / IDs]
    F --> H[Count overlay]
    G --> H
    H --> I[Annotated MP4]
    F --> J[JSON traffic summary]
~~~

## Repository structure

~~~text
verityagent-ai-vehicle-counter/
├── app.py
├── config.py
├── vehicle_detector.py
├── vehicle_tracker.py
├── vehicle_counter.py
├── video_processor.py
├── requirements.txt
├── .gitignore
├── README.md
├── LICENSE
├── tests/
│   ├── __init__.py
│   ├── test_vehicle_counter.py
│   ├── test_vehicle_tracker.py
│   └── test_video_processor.py
├── sample_videos/.gitkeep
└── outputs/.gitkeep
~~~

## Demo

No screenshot or GIF has been recorded yet. When you capture your own results, suggested assets are **docs/assets/vehicle-counter-preview.png** and **docs/assets/vehicle-counter-demo.gif**. These files are not included.

## Requirements

- Python 3.10–3.13 recommended.
- pip and a virtual environment.
- CPU and RAM suitable for the chosen model. A compatible PyTorch/GPU configuration may accelerate inference but is not required.
- A video readable by the installed OpenCV build and its codecs.
- Internet access on first use if model weights are not already cached.

## Installation

### Windows (PowerShell)

~~~powershell
git clone https://github.com/muhammadsaeed-ai/verityagent-ai-vehicle-counter.git
cd verityagent-ai-vehicle-counter
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

### macOS / Linux

~~~bash
git clone https://github.com/muhammadsaeed-ai/verityagent-ai-vehicle-counter.git
cd verityagent-ai-vehicle-counter
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Check the CLI and run unit tests:

~~~bash
python app.py --help
python -m pytest -q
~~~

The requirements pin Ultralytics 8.4.189 and OpenCV 4.14.0.94 and constrain NumPy to the range >=2.0.0 and <2.4. Availability can vary by Python version and platform. The standard requirements use opencv-python. On minimal servers, you may substitute a compatible opencv-python-headless build, but do not install both packages in the same environment because both provide the cv2 namespace. Re-run tests after changing OpenCV packages.

## Model setup and licensing

The default model is **yolo26n.pt**, a small pretrained Ultralytics YOLO Detect model. Ultralytics normally downloads the weights on first initialization if they are not cached; model weights are intentionally not committed here.

The detector checks model.names and resolves car, motorcycle, bus, and truck by name. For standard COCO-pretrained models these correspond to class IDs 2, 3, 5, and 7, respectively. The implementation does not assume those IDs for every model. If a custom model does not expose all four required category names, the app reports a clear error.

Choose another model with the **--model** option and a local weights path or supported Ultralytics model identifier. The selected model needs all four categories above for this project to run as configured.

**Third-party licensing matters.** This repository's source code is MIT-licensed. Ultralytics software and pretrained weights have their own licensing terms; the Ultralytics package advertises AGPL-3.0 and a separate Enterprise option. Review the terms for the exact release, model weights, and intended commercial use. OpenCV, PyTorch, and other dependencies also have separate licenses. This project's license does not relicense third-party components.

References: [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track), [Ultralytics package and license information](https://pypi.org/project/ultralytics/), [OpenCV Python package](https://pypi.org/project/opencv-python/).

## Usage

Place your own video at **sample_videos/traffic.mp4** or pass another input path. No sample traffic footage is bundled.

### Standard processing

~~~bash
python app.py --input sample_videos/traffic.mp4
~~~

Default outputs are **outputs/traffic_annotated.mp4** and **outputs/traffic_summary.json**.

### Explicit output paths

~~~bash
python app.py \
  --input sample_videos/traffic.mp4 \
  --output outputs/traffic_annotated.mp4 \
  --summary outputs/traffic_summary.json
~~~

### Headless processing

~~~bash
python app.py \
  --input sample_videos/traffic.mp4 \
  --output outputs/traffic_annotated.mp4 \
  --summary outputs/traffic_summary.json \
  --headless
~~~

### Configure line and confidence threshold

~~~bash
python app.py \
  --input sample_videos/traffic.mp4 \
  --line-y 300 \
  --confidence 0.35 \
  --direction both \
  --show-confidence
~~~

The horizontal line's Y coordinate is measured in pixels from the top of the frame and must fall inside the image. Omitting --line-y selects half the frame height. The application errors if a configured coordinate lies outside the input frame.

### Choose a model and direction

~~~bash
python app.py \
  --input sample_videos/traffic.mp4 \
  --model yolo26n.pt \
  --tracker bytetrack.yaml \
  --direction positive \
  --headless
~~~

Direction conventions for a horizontal line:

- **positive**: top-to-bottom motion (increasing image Y).
- **negative**: bottom-to-top motion (decreasing image Y).
- **both**: count both directions.

Use **python app.py --help** for all flags, including --fallback-fps and --verbose.

## Counting behavior

The bottom-center of each vehicle's bounding box is the anchor. The default hysteresis band is 8 pixels on either side of the line. Points inside this band do not change the remembered side, helping filter small fluctuations. A track must be observed on one stable side and then the other to establish a crossing. A vehicle appearing for one frame on the line is not counted.

Each track is counted **at most once per direction** in a video. With a direction filter, only the selected direction increments the summary. Detections without a track ID may be displayed but are excluded from counting.

## Output

### Annotated MP4

The output video preserves the input frame dimensions and draws the virtual line, supported vehicle boxes, class labels, track IDs when available, live per-class counts, and total crossings. Add **--show-confidence** to include confidence in labels.

### JSON summary

The JSON contains the input filename, configured model identifier, UTC ISO 8601 processing timestamp, processed frame count, valid source FPS or null, output FPS, frame dimensions, counts per class, total count, line location and direction convention, tracker, confidence threshold, output video path, headless setting, and measured processing duration.

If source FPS is unavailable or invalid, source_fps is null and the output writer uses --fallback-fps (25 FPS by default). The total is derived from category counts to keep figures consistent.

## Configuration

Defaults live in config.py and can be changed with CLI flags:

| Setting | Default | Argument |
|---|---|---|
| Model | yolo26n.pt | --model |
| Confidence | 0.35 | --confidence |
| Tracker | bytetrack.yaml | --tracker |
| Line Y | Half frame height | --line-y |
| Direction | both | --direction |
| Output video | outputs/traffic_annotated.mp4 | --output |
| JSON summary | outputs/traffic_summary.json | --summary |
| Invalid-FPS fallback | 25.0 | --fallback-fps |
| Confidence labels | Disabled | --show-confidence |

## Tests

The test suite covers line crossings, direction filters, duplicate suppression, totals, reset, missing IDs, tracker parsing and errors, invalid metadata, writer initialization failure, headless mode, resource cleanup, and JSON serialization.

~~~bash
python -m pytest -q
~~~

The tests mock the model and video boundaries, so no weights or video clip are needed. Passing unit tests does not prove inference works with every model, codec, or hardware setup.

## Limitations

- Occlusion, crowded scenes, lighting, poor resolution, camera movement, and perspective can reduce detection and tracking quality.
- Temporary tracking ID switches can cause missed or duplicated counts.
- The horizontal line assumes a reasonably stable camera and a suitable crossing location. Angled traffic may require different line geometry.
- The default model recognizes only categories supported by its training data and is not guaranteed to classify every scene correctly.
- Counts represent track crossings, not occupants, license-plate identities, or certified traffic measurements.
- Codec support, speed, and memory use depend on the system, model, hardware, and video. No real-time performance claim is made.

## Future improvements

- Real-time webcam input.
- Multiple counting lines.
- Vehicle speed estimation.
- Direction-specific traffic reports.
- CSV exports.
- Traffic dashboards.
- Region-of-interest filtering.
- Custom vehicle detection models.

These are possible extensions, not implemented features.

## License

The repository source code is released under the [MIT License](LICENSE). This does not override the terms applying to Ultralytics, its pretrained weights, OpenCV, PyTorch, or any other third-party dependency. Check the terms for each component before redistribution or commercial use.
