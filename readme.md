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

Dự án này triển khai hệ thống tracking đa đối tượng sử dụng YOLO11 detector và ByteTrack tracker:

- **MOT17 Dataset**: Đánh giá trên MOT17 benchmark với 4 checkpoint sizes (s, m, l, x)

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

### 1. GUI Tracking Application

```bash
python ByteTrack-YOLO/main.py
```

**Features:**
- Chọn video input
- Chọn vùng ROI (Region of Interest)
- Điều chỉnh tracking parameters
- Xem kết quả real-time
- Export tracking results

### 2. Test YOLO Detection

```bash
python test_yolo11s.py
```

### 3. Training YOLO (Jupyter Notebook)

```bash
jupyter notebook Train_Yolo.ipynb
```

## 📁 Cấu trúc dự án

```
KLTN/
├── � ByteTrack-YOLO/              # Main tracking application
│   ├── main.py                     # Entry point - GUI launcher
│   ├── app_gui.py                  # Tkinter GUI implementation
│   ├── model.py                    # YOLO model wrapper
│   ├── requirements.txt            # Dependencies
│   ├── configs/                    # Configuration files
│   ├── models/                     # Model checkpoints location
│   ├── outputs/                    # Detection outputs
│   └── src/
│       ├── detector/               # YOLO detector module
│       ├── tracker/                # ByteTrack tracker module
│       │   ├── bytetrack.py
│       │   ├── violation_detection.py
│       │   └── kalman_filter.py
│       └── utils/                  # Utility functions
│           ├── roi_utils.py
│           └── matching.py
│
├── 📂 benmark_result/              # Benchmark results
│   ├── benmark_bytetrack_Yolo.py  # Benchmark evaluation script
│   ├── results_archive_s_new/     # Small model results
│   ├── results_archive_m_new/     # Medium model results
│   ├── results_archive_l_new/     # Large model results
│   └── results_archive_x_new/     # Extra-large model results
│
├── 📄 .gitignore                   # Git ignore rules
├── 📄 README.md                    # This file
└── 📄 readme.md                    # Vietnamese documentation

```

## 📊 Kết quả

### ByteTrack MOT17 Tracking Results

Custom evaluation results on MOT17 dataset (from benmark_result archives):

| Model | MOTA ↑ | IDF1 ↑ | MOTP ↓ | Precision ↑ | Recall ↑ | ID Sw ↓ | FP ↓ | FN ↓ |
|-------|--------|--------|--------|-------------|----------|---------|--------|--------|
| bytetrack_s_mot17 | 81.0% | 81.9% | 0.662 | 95.9% | 85.3% | 379 | 3431 | 12449 |
| bytetrack_m_mot17 | 84.2% | 85.2% | 0.633 | 92.7% | 92.5% | 361 | 9829 | 5834 |
| bytetrack_l_mot17 | 84.6% | 85.7% | 0.616 | 91.2% | 94.7% | 410 | 11418 | 4008 |
| bytetrack_x_mot17 | 86.5% | 86.2% | 0.577 | 93.4% | 94.0% | 355 | 6844 | 4649 |

**Best Model**: bytetrack_x_mot17 achieves highest MOTA (86.5%) with best MOTP (0.577)

### Traffic Tracking (YOLO11s + ByteTrack)

- **Classes**: Car, Truck, Bus, Motorbike, Bicycle, Person
- **Performance**: ~25-30 FPS on RTX 3060
- **Recommended Model**: bytetrack_x_mot17 (best overall performance)

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

## 📝 Dataset (Không bao gồm trong repo)

### MOT17 Dataset

MOT17 dataset được sử dụng cho đánh giá performance của ByteTrack tracker.

**Download**: [MOT Challenge Website](https://motchallenge.net/data/MOT17/)

**Dataset Structure**:
```
Dataset/MOT17/
├── train/
│   ├── MOT17-02/
│   ├── MOT17-04/
│   ├── MOT17-05/
│   ├── MOT17-09/
│   ├── MOT17-10/
│   ├── MOT17-11/
│   └── MOT17-13/
└── test/
    └── [test sequences]
```


## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **ByteTrack**: [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
- **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **MOT Challenge**: [motchallenge.net](https://motchallenge.net/)

## 📧 Contact

- Email: tranthaidaiduong0@gmail.com
- GitHub: [@yourusername](https://github.com/duong0410)

---

**⭐ If you find this project helpful, please give it a star!**

---

## 📝 Phụ lục: Dataset Processing Notes

<details>
<summary>COCO & UA-DETRAC Dataset Processing (Click to expand)</summary>

### CLASSES:


\# Đọc file annotation COCO.

\# id2name: map từ category\_id → tên class.

\# Lọc annotation chỉ giữ các class nằm trong danh sách CLASSES.

\# imgs: chứa thông tin từng ảnh (file\_name, width, height...).

\# per\_img: gom annotation theo image\_id để dễ xử lý.

\# Trả về (imgs, per\_img, id2name) phục vụ việc copy ảnh + tạo nhãn YOLO.



def load\_ua\_with\_classes(label\_folder):

\# Quét toàn bộ file .txt chứa bbox của UA-DETRAC.

\# Đọc từng dòng, kiểm tra class\_id có trong UA\_MAP.

\# Map class theo UA\_MAP → class tên chuẩn.

\# Lưu lớp (class) mà mỗi ảnh chứa vào:

\#     img\_classes\[image\_stem] = {set các class trong ảnh}

\# Dùng để biết ảnh UA thuộc lớp nào khi sampling theo class.



def get\_coco\_img\_classes(per\_img, id2name):

\# Duyệt annotation của từng ảnh COCO.

\# Lấy tên class từ category\_id bằng id2name.

\# Lọc chỉ giữ các class nằm trong CLASSES.

\# Lưu các class xuất hiện trong mỗi ảnh vào:

\#      img\_classes\[img\_id] = {set class trong ảnh}

\# Dùng để phân loại ảnh COCO theo class.





def balanced\_sampling()



\# Hàm balanced\_sampling() tạo ra danh sách ảnh được chọn sao cho cân bằng giữa hai nguồn

(UA-DETRAC và COCO), đồng thời đảm bảo mỗi class có số lượng ảnh nằm trong khoảng

min\_target – max\_target và tỉ lệ giữa hai nguồn đúng với ratio\_ua.





\# Chuẩn bị dữ liệu: Hàm nhận vào hai dict: ua\_img\_classes và coco\_img\_classes, mỗi dict chứa img\_id và danh sách class xuất hiện trong ảnh đó. Tất cả ảnh từ hai nguồn được gom vào all\_images, mỗi ảnh có thông tin: nguồn (ua/coco), id, các class của ảnh và num\_classes (số class).





\# Nhóm ảnh theo số lượng class: Ảnh được chia thành 3 nhóm dựa trên số class trong ảnh: 1 class, 2 class hoặc ≥3 class. Điều này là bởi vì vì ảnh 1-class là nguồn tốt nhất để điều chỉnh số lượng class.





\# Chuẩn bị cấu trúc chứa ảnh theo từng class: class\_images\[class]\[num\_classes] chứa danh sách ảnh thuộc class đó và nhóm num\_classes. Nhờ đó có thể truy cập nhanh ảnh cần dùng cho từng bước và từng class.





\# PHASE 1 – Chọn ảnh 1-class theo tỉ lệ UA/COCO: Với mỗi class nếu class đó đều có dữ liệu ở cả 2 bộ UA và COCO thì chọn ảnh 1-class từ UA và COCO sao cho đúng với tỉ lệ ratio\_ua. Ví dụ ratio\_ua = 0.7 → 70% ảnh đến từ UA và 30% từ COCO. Ảnh 1-class được ưu tiên vì chúng giúp kiểm soát từng class một cách chính xác.





\# PHASE 2 – Oversampling ảnh 1-class nếu chưa đủ target: Sau Phase 1, nếu class vẫn chưa đạt target, thực hiện oversampling (nhân bản ảnh). Ảnh 1-class bất kỳ được chọn ngẫu nhiên để nhân lên cho đến khi đạt target. Mỗi bản sao được gắn oversample\_id để phân biệt.





\# PHASE 3 – Giảm số lượng nếu vượt quá max\_target (loại ảnh 1-class trước): Nếu số lượng ảnh của class vượt max\_target, ưu tiên loại bỏ ảnh 1-class. Ảnh 1-class chỉ thuộc đúng một class nên loại bỏ chúng sẽ không ảnh hưởng class khác.





\# PHASE 4 – Nếu vẫn dư: loại ảnh 2-class: Nếu loại ảnh 1-class mà dữ liệu còn lại vẫn chưa giảm xuống dưới max\_target, mới xét loại ảnh 2-class. Tuy nhiên chỉ loại khi class còn lại trong ảnh đó vẫn > min\_target để giữ cân bằng.





\# Tách kết quả: Cuối cùng gom ảnh đã chọn thành hai danh sách:

   selected\_ua   – các ảnh từ UA

   selected\_coco – các ảnh từ COCO

  Mỗi phần tử có dạng (img\_id, oversample\_id). oversample\_id = None → ảnh gốc.





\# Xuất thống kê và trả về: Hàm in ra phân phối class cuối cùng và trả về selected\_ua và selected\_coco để dùng cho bước copy ảnh thực tế.





def copy\_ua\_image\_label(...)



\# Kiểm tra ảnh + label gốc có tồn tại không → nếu thiếu trả về False.

\# Tạo tên file mới, thêm hậu tố \_oversample\_id nếu cần.

\# Copy ảnh sang thư mục đích.

\# Đọc label của UA-DETRAC, ánh xạ class bằng UA\_MAP, loại class không dùng.

\# Chuyển class → id YOLO dựa trên cls\_to\_id.

\# Ghi label sang format YOLO:

 	class cx cy w h

\# Trả về True nếu mọi thứ ok.





def copy\_coco\_image\_label(...)



\#Kiểm tra img\_id có trong COCO không → nếu không trả về False.

\#Lấy thông tin ảnh (file\_name, width, height).

\#Tạo tên file đích, thêm hậu tố nếu oversample.

\#Copy ảnh sang thư mục đích.

\#Tạo đường dẫn label .txt.

\#Lấy danh sách annotation của ảnh (coco\_perimg).

\#Convert bbox từ COCO → YOLO bằng coco\_to\_yolo().

\#Ghi label theo format YOLO:

 	class cx cy w h

\#Trả về True nếu xử lý xong.





**TỔNG QUAN:  Đoạn code này dùng để:**



* **Trộn 2 dataset:**
* 
* 

\*\* 	UA-DETRAC (ảnh + label dạng YOLO)\*\*



\*\*COCO 2017 (ảnh + annotation COCO JSON)\*\*







* **Chỉ lấy các lớp: car, truck, bus, van, motorcycle, bicycle, person.**



* **Tạo dataset cân bằng theo từng lớp:**



  **Mỗi lớp cần khoảng TARGET\_TRAIN (2500) ảnh cho train**

  

  **Và TARGET\_VAL (500) ảnh cho val**

  

  **Có khoảng cho phép (min/max target)**

  

* **Thực hiện 4 giai đoạn:**

  

  **Phase 1: Chọn ảnh single-class theo đúng tỷ lệ UA/COCO (70/30)**

  

  **Phase 2: Oversample nếu lớp nào thiếu ảnh (nhân bản nhiều lần)**

  

  **Phase 3: Loại bớt ảnh single-class nếu lớp nào vượt quá max\_target**

  

  **Phase 4: Nếu vẫn thừa → loại thêm ảnh 2-class**

  

  **Cuối cùng copy ảnh + tạo nhãn YOLO mới vào thư mục mixed\_dataset\_balanced.**

