# Hướng dẫn Stratified K-Fold Cross-Validation

## Tổng quan

Script này cho phép bạn chạy **Stratified K-Fold Cross-Validation** cho moBRCA-net, giúp đánh giá model một cách đáng tin cậy hơn so với train/test split đơn giản.

## Hai phiên bản model

1. **`moBRCA-net.py`**: Phiên bản đầy đủ với **Contrastive Learning** pretraining
   - Script: `run_kfold.py`
   - Kết quả lưu trong: `kfold_results/`

2. **`moBRCA-net_baseline.py`**: Phiên bản baseline **không có** Contrastive Learning
   - Script: `run_kfold_baseline.py`
   - Kết quả lưu trong: `kfold_results_baseline/`

## Các bước thực hiện

### Lưu ý: Có thể dùng chung dữ liệu k-fold cho cả 2 phiên bản

Cả hai phiên bản (với và không có contrastive learning) có thể dùng chung cùng một bộ folds. Bạn chỉ cần chạy `prepare_kfold_data.py` một lần.

### Bước 1: Chuẩn bị dữ liệu cho K-Fold

Chạy script `prepare_kfold_data.py` để chia dữ liệu thành k folds:

```powershell
python prepare_kfold_data.py `
    --label-path data/54814634_BRCA_label_num.csv `
    --label-column Label `
    --zscore `
    --output-dir ./kfold_output `
    --k-folds 5 `
    --top-gene 1000 --top-cpg 1000 --top-mirna 100 `
    --seed 42
```

**Tham số:**
- `--k-folds`: Số lượng folds (mặc định: 5)
- `--output-dir`: Thư mục lưu kết quả
- `--seed`: Random seed để đảm bảo reproducibility
- Các tham số khác giống `prepare_data.py`

**Kết quả:**
- Thư mục `folds/fold_1/`, `folds/fold_2/`, ... chứa train/test split cho từng fold
- File `feature_counts.txt` với số lượng features

### Bước 2: Chạy K-Fold Cross-Validation

#### 2a. Với Contrastive Learning (moBRCA-net.py)

Chạy script `run_kfold.py` để train model trên từng fold:

```powershell
# Thiết lập biến môi trường (tùy chọn)
$env:EPOCHS=50
$env:BATCH_SIZE=64
$env:LR=1e-2

# Chạy k-fold CV
python run_kfold.py `
    --base-dir ./kfold_output `
    --k-folds 5 `
    --epochs 50 `
    --batch-size 64 `
    --lr 1e-2
```

**Tham số:**
- `--base-dir`: Thư mục chứa folds và feature_counts.txt (output của bước 1)
- `--k-folds`: Số lượng folds (phải khớp với bước 1)
- `--epochs`: Số epochs training (hoặc dùng EPOCHS env var)
- `--batch-size`: Batch size (hoặc dùng BATCH_SIZE env var)
- `--lr`: Learning rate (hoặc dùng LR env var)
- `--dropout`: Dropout rate (mặc định: 0.2)
- `--weight-decay`: Weight decay (hoặc dùng WEIGHT_DECAY env var)

#### 2b. Baseline - Không có Contrastive Learning (moBRCA-net_baseline.py)

Chạy script `run_kfold_baseline.py`:

```powershell
# Thiết lập biến môi trường (tùy chọn)
$env:EPOCHS=50
$env:BATCH_SIZE=64
$env:LR=1e-2

# Chạy k-fold CV cho baseline
python run_kfold_baseline.py `
    --base-dir ./kfold_output `
    --k-folds 5 `
    --epochs 50 `
    --batch-size 64 `
    --lr 1e-2
```

### Bước 3: Chạy tự động (Khuyến nghị)

#### 3a. Với Contrastive Learning

**Windows PowerShell:**
```powershell
.\run_kfold_cv.ps1
```

**Linux/Mac/Git Bash:**
```bash
bash run_kfold_cv.sh
```

#### 3b. Baseline

**Windows PowerShell:**
```powershell
.\run_kfold_cv_baseline.ps1
```

**Linux/Mac/Git Bash:**
```bash
bash run_kfold_cv_baseline.sh
```

Bạn có thể chỉnh sửa các tham số trong các script này trước khi chạy.

## Kết quả

### Với Contrastive Learning
Sau khi chạy xong, kết quả được lưu trong `{output_dir}/kfold_results/`:

### Baseline (không có Contrastive Learning)
Kết quả được lưu trong `{output_dir}/kfold_results_baseline/`:

### Cấu trúc thư mục:
```
kfold_output/
├── folds/
│   ├── fold_1/
│   │   ├── train_X.csv
│   │   ├── train_Y.csv
│   │   ├── test_X.csv
│   │   └── test_Y.csv
│   ├── fold_2/
│   └── ...
├── feature_counts.txt
└── kfold_results/
    ├── fold_1/
    │   ├── prediction.csv
    │   ├── label.csv
    │   ├── attn_score_gene.csv
    │   ├── attn_score_methyl.csv
    │   └── attn_score_mirna.csv
    ├── fold_2/
    ├── ...
    ├── kfold_summary.csv          # Tổng hợp metrics (mean, std, min, max)
    ├── per_fold_metrics.csv       # Metrics chi tiết từng fold
    ├── all_predictions.csv        # Tất cả predictions (pooled)
    ├── all_labels.csv             # Tất cả labels (pooled)
    └── overall_confusion_matrix.csv
```

### Các file kết quả:

1. **`kfold_summary.csv`**: 
   - Mean, Std, Min, Max của Accuracy, Precision, Recall, F1-Score
   - Đánh giá tổng quan về performance của model

2. **`per_fold_metrics.csv`**:
   - Metrics chi tiết cho từng fold
   - Giúp phân tích độ ổn định của model

3. **`all_predictions.csv`** và **`all_labels.csv`**:
   - Tất cả predictions và labels từ tất cả folds (pooled)
   - Dùng để tính overall metrics và confusion matrix

4. **`overall_confusion_matrix.csv`**:
   - Confusion matrix tính trên tất cả predictions

## Ví dụ Output

```
STRATIFIED K-FOLD CROSS-VALIDATION
================================================================================
Base directory: ./kfold_output
Number of folds: 5
Features: gene=1000, cpg=1000, mirna=100
Classes: 5
Epochs: 50, Batch size: 64, LR: 0.01
================================================================================

================================================================================
FOLD 1/5
================================================================================
...
📊 Fold 1 Results:
   Accuracy:  0.8235
   Precision: 0.8100
   Recall:    0.8150
   F1-Score:  0.8125

...

================================================================================
K-FOLD CROSS-VALIDATION SUMMARY
================================================================================

     Metric       Mean      Std       Min       Max
0  Accuracy   0.820000  0.012247  0.810000  0.835000
1 Precision   0.815000  0.015811  0.800000  0.835000
2    Recall   0.818000  0.011402  0.805000  0.830000
3  F1-Score   0.816500  0.013578  0.802500  0.832500

📊 Overall Metrics (pooled across all folds):
   Accuracy:  0.8210
   Precision: 0.8165
   Recall:    0.8185
   F1-Score:  0.8175
```

## So sánh với Train/Test Split

| Đặc điểm | Train/Test Split | K-Fold CV |
|----------|------------------|-----------|
| Số lần train | 1 lần | k lần (ví dụ: 5) |
| Dữ liệu test | Cố định 20% | Mỗi fold test khác nhau |
| Độ tin cậy | Thấp hơn | Cao hơn |
| Thời gian | Nhanh | Chậm hơn (k lần) |
| Đánh giá | Một lần | Kết quả trung bình + std |

## Lưu ý

1. **Thời gian chạy**: K-fold CV sẽ chạy lâu hơn k lần so với train/test split đơn giản
2. **Memory**: Mỗi fold train độc lập nên không tốn thêm memory so với train/test split
3. **Reproducibility**: Dùng `--seed` để đảm bảo kết quả có thể reproduce
4. **Số lượng folds**: Thường dùng 5 hoặc 10 folds. Với dataset nhỏ (< 200 samples), nên dùng 5 folds. Với dataset lớn, có thể dùng 10 folds.

## Troubleshooting

**Lỗi: "No results found for fold X"**
- Kiểm tra xem fold directory có tồn tại không
- Kiểm tra xem training có chạy thành công trên fold đó không

**Lỗi: "Feature counts file not found"**
- Đảm bảo đã chạy `prepare_kfold_data.py` trước
- Kiểm tra đường dẫn `--base-dir`

**Lỗi: Out of memory**
- Giảm `--batch-size` xuống (ví dụ: 32 hoặc 16)
- Giảm số lượng features (`--top-gene`, `--top-cpg`, `--top-mirna`)

