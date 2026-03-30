# 🚗 Multi-Object Tracking System - KLTN Project

Hệ thống theo dõi đa đối tượng (Multi-Object Tracking) sử dụng YOLO detector và ByteTrack tracker cho bài toán theo dõi phương tiện giao thông và người đi bộ.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Kết quả](#kết-quả)
- [Công nghệ](#công-nghệ)

## 🎯 Tổng quan

Dự án này triển khai hệ thống tracking đa đối tượng cho hai bài toán chính:

1. **Traffic Tracking**: Theo dõi các phương tiện giao thông

### Trackers được triển khai:

- ✅ **ByteTrack** (custom implementation)

## ✨ Tính năng

### 🖥️ GUI Application

- **GUI tracking video** với YOLO11s + ByteTrack
- ROI selection (chọn vùng quan tâm)
- Real-time visualization
- Track filtering và statistics
- Export tracking results

### 📊 Video Tracking

- **Real-time tracking** với ByteTrack
- Metrics: MOTA, IDF1, MOTP, Precision, Recall, ID Switches, Fragmentations


### 🎯 Detection Models

- **YOLO11s** (Ultralytics YOLO11)
- Custom trained on traffic datasets

## 🔧 Cài đặt

### Yêu cầu hệ thống

- Python 3.10+
- CUDA 11.8+ (nếu dùng GPU)
- 8GB+ RAM
- 4GB+ VRAM (GPU)

### Cài đặt thư viện

```bash
# Clone repository
git clone https://github.com/yourusername/KLTN.git
cd KLTN

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

### requirements.txt

```txt
# Deep Learning & Computer Vision
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0

# Tracking Libraries
boxmot>=10.0.0
filterpy>=1.4.5

# MOT Evaluation
motmetrics>=1.4.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0

# Utilities
tqdm>=4.65.0
PyYAML>=6.0
```

## 🚀 Sử dụng

```bash
python ByteTrack-YOLO/main.py
```

**Features:**
- Chọn video input
- Chọn vùng ROI (Region of Interest)
- Điều chỉnh tracking parameters
- Xem kết quả real-time
- Export tracking results

## 📁 Cấu trúc dự án

```
KLTN/
├── 📄 bytetrack_test.py           # GUI tracking application
├── 📄 yolo11_bytetrack.py        # YOLO + ByteTrack integration
├── 📄 test_yolo11s.py            # YOLO detection testing
├── 📄 Train_Yolo.ipynb           # Training notebook
│
├── 📂 ByteTrack-YOLO/            # ByteTrack custom implementation
│   ├── main.py
│   ├── model.py
│   ├── run_gui.py
│   └── src/
│
├── 📂 Detector_train/            # Dataset preparation scripts
│   ├── check_dataset_paths.py
│   └── load-data-coco-ua-detrac.ipynb
│
├── 📂 Dataset/                   # ⚠️ NOT INCLUDED IN GIT
│   ├── MOT17/                    # MOT17 dataset
│   ├── traffic_yolo/             # Traffic dataset v1
│   ├── traffic_yolo_v2/          # Traffic dataset v2
│   ├── traffic_yolo_v3/          # Traffic dataset v3
│   ├── COCO2017/                 # COCO dataset
│   └── UA-DETRAC/                # UA-DETRAC dataset
│
├── 📄 .gitignore                 # Git ignore rules
├── 📄 README.md                  # This file
├── 📄 requirements.txt           # Python dependencies
└── 📄 readme.txt                 # Vietnamese notes

```

## 📊 Kết quả

### ByteTrack MOT17 Tracking Results

Evaluation results on MOT17 dataset train using detection checkpoint from origin paper:

| Model | MOTA ↑ | IDF1 ↑ | MOTP ↓ | Precision ↑ | Recall ↑ | ID Sw ↓ | FP ↓ | FN ↓ |
|-------|--------|--------|--------|-------------|----------|---------|--------|--------|
| bytetrack_s_mot17 | 81.0% | 81.9% | 0.662 | 95.9% | 85.3% | 379 | 3431 | 12449 |
| bytetrack_m_mot17 | 84.2% | 85.2% | 0.633 | 92.7% | 92.5% | 361 | 9829 | 5834 |
| bytetrack_l_mot17 | 84.6% | 85.7% | 0.616 | 91.2% | 94.7% | 410 | 11418 | 4008 |
| bytetrack_x_mot17 | 86.5% | 86.2% | 0.577 | 93.4% | 94.0% | 355 | 6844 | 4649 |

**Best Model**: bytetrack_x_mot17 achieves highest MOTA (86.5%) with best MOTP (0.577)


## 🛠️ Công nghệ

### Deep Learning Frameworks
- **PyTorch** 2.0+
- **Ultralytics YOLO** 11

### Tracking Algorithms
- **ByteTrack**: 2-phase matching strategy
- **Kalman Filter**: Motion prediction

### Evaluation Metrics
- **MOT Metrics** (via motmetrics)
  - MOTA (Multi-Object Tracking Accuracy)
  - IDF1 (ID F1 Score)
  - MOTP (Multi-Object Tracking Precision)
  - Precision, Recall
  - ID Switches, Fragmentations

## 📝 Dataset Setup (Không bao gồm trong repo)

### MOT17 Dataset

MOT17 dataset được sử dụng cho đánh giá performance của ByteTrack tracker.

**Download**: [MOT Challenge Website](https://motchallenge.net/data/MOT17/)

**Dataset Structure**:
```
Dataset/MOT17/
├── train/
│   ├── MOT17-02-DPM/
│   ├── MOT17-02-FRCNN/
│   ├── MOT17-02-SDP/
│   └── ...
└── test/
    └── ...
```

**Download**: [MOT Challenge](https://motchallenge.net/data/MOT17/)

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **ByteTrack**: [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
- **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **MOT Challenge**: [motchallenge.net](https://motchallenge.net/)

## 📧 Contact

- Email: tranthaidaiduong0@gmail.com
- GitHub: [@duong0410](https://github.com/duong0410)

---

**⭐ If you find this project helpful, please give it a star!**

---
