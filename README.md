# Lab 3: UrbanSound8K — 1D CNN for Environmental Sound Classification

> **Môn học:** Học sâu (CSC4005)  
> **Họ tên:** Trần Việt Vinh  
> **MSSV:** 1771040030

---

## 📁 Cấu trúc repo

```
csc4005-lab3-1dcnn-TRAN-VIET-VINH-1771040030/
├── README.md
├── REPORT_LAB3.md
├── requirements.txt
├── configs/
│   ├── baseline_mfcc_1dcnn.json
│   ├── fast_debug.json
│   └── extension_raw_waveform.json
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── data/
│   └── UrbanSound8K/               ← Dữ liệu (không push lên GitHub)
│       ├── audio/
│       └── metadata/
├── outputs/
│   └── 1771040030_mfcc_baseline/
│       ├── best_model.pt
│       ├── history.csv
│       ├── curves.png
│       ├── confusion_matrix.png
│       └── metrics.json
└── wandb/
```

---

## 🛠️ Cài đặt môi trường

### 1. Tạo môi trường Conda

```bash
conda create -n csc4005-dl python=3.10 -y
conda activate csc4005-dl
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3. Đăng nhập W&B

```bash
wandb login
# Dán API key từ https://wandb.ai/authorize
```

---

## 📂 Chuẩn bị dữ liệu

1. Tải dataset **UrbanSound8K** theo đường dẫn trong đề bài.
2. Giải nén và đặt vào thư mục `data/` theo cấu trúc sau:

```
data/
└── UrbanSound8K/
    ├── audio/
    │   ├── fold1/
    │   ├── fold2/
    │   └── ...fold10/
    └── metadata/
        └── UrbanSound8K.csv
```

> ⚠️ **Lưu ý:** Không push thư mục `data/` lên GitHub (đã có trong `.gitignore`).

---

## 🚀 Chạy huấn luyện

### 🔹 Debug — kiểm tra pipeline nhanh

```bash
python -m src.train \
  --config configs/fast_debug.json \
  --data_dir ./data/UrbanSound8K
```

### 🔹 Baseline chính — MFCC + 1D-CNN

```bash
python -m src.train \
  --config configs/baseline_mfcc_1dcnn.json \
  --data_dir ./data/UrbanSound8K \
  --run_name "1771040030_mfcc_baseline"
```

### 🔹 (Tùy chọn) Log-mel features

```bash
python -m src.train \
  --config configs/baseline_mfcc_1dcnn.json \
  --data_dir ./data/UrbanSound8K \
  --feature_type logmel \
  --run_name "1771040030_logmel"
```

### 🔹 (Mở rộng) Raw waveform

```bash
python -m src.train \
  --config configs/extension_raw_waveform.json \
  --data_dir ./data/UrbanSound8K \
  --run_name "1771040030_raw_waveform"
```

---

## 📊 Kết quả đạt được

### Baseline — MFCC + 1D-CNN

| Chỉ số | Giá trị |
|--------|---------|
| Best validation accuracy | **59.61%** |
| Test accuracy | **52.47%** |
| Thời gian trung bình / epoch | 5.14 giây |
| Số lượng tham số | 137,930 |
| Số epochs | 11 / 12 *(early stopping)* |

### Kết quả theo lớp

**Top 3 lớp tốt nhất:**

| Hạng | Lớp | Accuracy |
|------|-----|----------|
| 🥇 | `gun_shot` | 100.0% |
| 🥈 | `siren` | 93.9% |
| 🥉 | `street_music` | 93.3% |

**Lớp khó phân loại nhất:**

| Lớp | Accuracy | Nhầm sang chủ yếu |
|-----|----------|--------------------|
| `drilling` | 45.8% | `jackhammer` *(6 lần)* |
| `dog_bark` | 66.7% | `car_horn` *(3 lần)* |
| `children_playing` | 71.4% | `siren` *(4 lần)* |

---

## 📈 Outputs sau khi huấn luyện

Kết quả được tự động lưu vào thư mục `outputs/<run_name>/`:

```
outputs/1771040030_mfcc_baseline/
├── best_model.pt        # Model checkpoint tốt nhất (theo val accuracy)
├── history.csv          # Loss & accuracy theo từng epoch
├── curves.png           # Learning curves (train/val)
├── confusion_matrix.png # Confusion matrix trên tập test
└── metrics.json         # Tổng hợp metrics cuối cùng
```

---

## 🔗 W&B Dashboard

| | Link |
|-|------|
| **Project** | [csc4005-lab3-urbansound-1dcnn](https://wandb.ai/vinhtran2785-bt/csc4005-lab3-urbansound-1dcnn) |
| **Run baseline** | [1771040030_mfcc_baseline](https://wandb.ai/vinhtran2785-bt/csc4005-lab3-urbansound-1dcnn/runs/h3ehkdo7) |

> ⚠️ Cần đăng nhập tài khoản `vinhtran2785` để xem dashboard.

---

## 📝 Ghi chú

- Xem phân tích chi tiết, confusion matrix và trả lời câu hỏi tự kiểm tra tại [`REPORT_LAB3.md`](./REPORT_LAB3.md).
- Để thêm run mới, đặt `--run_name` khác với tên đã có để tránh ghi đè checkpoint.
- Nếu gặp lỗi `CUDA out of memory`, giảm `batch_size` trong file config tương ứng.