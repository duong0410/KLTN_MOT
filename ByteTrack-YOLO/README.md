# ByteTrack-YOLO: Multi-Object Tracking System

A professional vehicle tracking system combining ByteTrack algorithm with YOLO11s detector. Features class-persistent tracking to prevent ID/class switching during occlusion.

## 🌟 Features

- **ByteTrack Algorithm**: Two-phase association for robust tracking
- **Class Persistence**: Once assigned, object class never changes
- **YOLO11s Integration**: Fast and accurate vehicle detection
- **GUI Application**: User-friendly interface for video processing
- **Configurable**: YAML-based configuration system
- **Modular Design**: Clean, maintainable code structure

## 📁 Project Structure

```
ByteTrack-YOLO/
├── src/
│   ├── tracker/          # ByteTrack implementation
│   │   └── bytetrack.py
│   ├── detector/         # YOLO detector
│   │   └── yolo_detector.py
│   ├── utils/           # Utility functions
│   │   ├── kalman_filter.py
│   │   ├── matching.py
│   │   ├── bbox.py
│   │   └── visualization.py
│   └── gui/             # GUI application
│       └── app.py
├── configs/             # Configuration files
│   └── default_config.yaml
├── models/              # YOLO models (place your .pt files here)
├── outputs/             # Output videos
├── tests/               # Unit tests
├── docs/                # Documentation
├── requirements.txt     # Python dependencies
├── setup.py            # Package setup
├── main.py             # CLI entry point
├── run_gui.py          # GUI entry point
└── README.md           # This file
```

## 🚀 Installation

### 1. Clone the repository

```bash
cd ByteTrack-YOLO
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO model

Place your trained YOLO11s model in `models/` directory:
```
models/yolo11s_traffic.pt
```

## 💻 Usage

### GUI Application (Recommended)

```bash
python run_gui.py
```

Features:
- Browse and select video files
- Adjust tracking parameters in real-time
- Preview tracking results
- Save processed videos

### Command Line Interface

Process a video:
```bash
python main.py --video path/to/video.mp4 --output outputs/result.mp4
```

With custom configuration:
```bash
python main.py --video path/to/video.mp4 --config configs/custom_config.yaml
```

### Python API

```python
from src.detector import YOLODetector
from src.tracker import BYTETracker
import cv2

# Initialize detector and tracker
detector = YOLODetector(
    model_path='models/yolo11s_traffic.pt',
    conf_threshold=0.01,
    device='cuda'
)

tracker = BYTETracker(
    det_conf_high=0.5,
    det_conf_low=0.1,
    new_track_thresh=0.6
)

# Process video
cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect
    detections = detector.detect(frame)
    
    # Track
    tracks = tracker.update(detections, frame.shape[:2])
    
    # Use tracks...
```

## ⚙️ Configuration

Edit `configs/default_config.yaml` to customize parameters:

```yaml
detection:
  model_path: "models/yolo11s_traffic.pt"
  conf_threshold: 0.01
  device: "cuda"

tracker:
  det_conf_high: 0.5
  det_conf_low: 0.1
  new_track_thresh: 0.6
  match_thresh_high: 0.8
  match_thresh_low: 0.5
  track_buffer: 30
  min_hits: 1
```

## 🔧 Key Parameters

### Detection
- `conf_threshold`: Minimum confidence for YOLO detection (default: 0.01)

### Tracking
- `det_conf_high`: High confidence threshold (default: 0.5)
- `det_conf_low`: Low confidence threshold (default: 0.1)
- `new_track_thresh`: Threshold for creating new tracks (default: 0.6)
- `match_thresh_high`: First association threshold (default: 0.8, IoU > 0.2)
- `match_thresh_low`: Second association threshold (default: 0.5, IoU > 0.5)
- `track_buffer`: Frames to keep lost tracks (default: 30)
- `min_hits`: Minimum hits before confirmation (default: 1)

## 📊 Algorithm Overview

### ByteTrack Two-Phase Association

1. **Phase 1**: Match existing tracks with high-confidence detections
2. **Phase 2**: Match remaining tracks with low-confidence detections
3. **New Tracks**: Create new tracks from unmatched high-confidence detections
4. **Lost Tracks**: Remove tracks that haven't been matched for `track_buffer` frames

### Class Persistence

- Each track stores its class ID upon creation
- Class ID **never changes** during tracking lifetime
- Prevents class switching during occlusion or detection errors

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

## 📝 Citation

ByteTrack paper:
```bibtex
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}
```

## 📄 License

This project is for educational purposes. Please cite the original ByteTrack paper if you use this code in your research.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- ByteTrack original implementation
- Ultralytics YOLO
- MOT Challenge dataset

---

**Note**: This project requires a trained YOLO11s model. Make sure to place your model file in the `models/` directory before running.
