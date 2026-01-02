# Hướng dẫn chạy thủ công K-Fold Cross-Validation

## Bước 1: Chuẩn bị dữ liệu

```powershell
cd D:\semester2025.1\deep_learning\project\source_code_paper2023\deep-learning-omics

python prepare_kfold_data.py --label-path data/54814634_BRCA_label_num.csv --label-column Label --zscore --output-dir ./kfold_output --k-folds 5 --top-gene 500 --top-cpg 500 --top-mirna 100 --seed 42
```

## Bước 2: Chạy K-Fold CV

### Model có Contrastive Learning:

```powershell
$env:EPOCHS=50; $env:BATCH_SIZE=64; $env:LR=1e-2; python run_kfold.py --base-dir ./kfold_output --k-folds 5 --epochs 50 --batch-size 64 --lr 1e-2
```

### Model Baseline:

```powershell
$env:EPOCHS=50; $env:BATCH_SIZE=64; $env:LR=1e-2; python run_kfold_baseline.py --base-dir ./kfold_output --k-folds 5 --epochs 50 --batch-size 64 --lr 1e-2
```

## Hoặc chạy từng bước chi tiết:

### Bước 2a: Thiết lập biến môi trường

```powershell
$env:EPOCHS=50
$env:BATCH_SIZE=64
$env:LR=1e-2
$env:WEIGHT_DECAY=1e-4
```

### Bước 2b: Chạy training

**Model có Contrastive Learning:**
```powershell
python run_kfold.py --base-dir ./kfold_output --k-folds 5 --epochs 50 --batch-size 64 --lr 1e-2
```

**Model Baseline:**
```powershell
python run_kfold_baseline.py --base-dir ./kfold_output --k-folds 5 --epochs 50 --batch-size 64 --lr 1e-2
```

## Tham số có thể điều chỉnh:

### Trong prepare_kfold_data.py:
- `--top-gene`: Số lượng gene features (mặc định: tất cả, ví dụ: 500, 1000)
- `--top-cpg`: Số lượng CpG features (mặc định: tất cả, ví dụ: 500, 1000)
- `--top-mirna`: Số lượng miRNA features (mặc định: tất cả, ví dụ: 100, 200)
- `--k-folds`: Số lượng folds (mặc định: 5)
- `--seed`: Random seed (mặc định: 42)
- `--zscore`: Có/không chuẩn hóa z-score (thêm flag để bật)

### Trong run_kfold.py / run_kfold_baseline.py:
- `--epochs`: Số epochs training (mặc định: 200, hoặc từ env EPOCHS)
- `--batch-size`: Batch size (mặc định: 136, hoặc từ env BATCH_SIZE)
- `--lr`: Learning rate (mặc định: 1e-2, hoặc từ env LR)
- `--dropout`: Dropout rate (mặc định: 0.2)
- `--weight-decay`: Weight decay (mặc định: 1e-4, hoặc từ env WEIGHT_DECAY)

## Ví dụ với các tham số khác:

### Ví dụ 1: Chạy với 1000 genes, 1000 CpG, 200 miRNAs, 10 folds
```powershell
# Bước 1
python prepare_kfold_data.py --label-path data/54814634_BRCA_label_num.csv --label-column Label --zscore --output-dir ./kfold_output --k-folds 10 --top-gene 1000 --top-cpg 1000 --top-mirna 200 --seed 42

# Bước 2
python run_kfold.py --base-dir ./kfold_output --k-folds 10 --epochs 100 --batch-size 128 --lr 0.01
```

### Ví dụ 2: Chạy với ít features hơn (300, 300, 50)
```powershell
# Bước 1
python prepare_kfold_data.py --label-path data/54814634_BRCA_label_num.csv --label-column Label --zscore --output-dir ./kfold_output --k-folds 5 --top-gene 300 --top-cpg 300 --top-mirna 50 --seed 42

# Bước 2
python run_kfold.py --base-dir ./kfold_output --k-folds 5 --epochs 50 --batch-size 32 --lr 0.01
```

## Kiểm tra kết quả:

Sau khi chạy xong, kiểm tra:
```powershell
# Xem feature counts
Get-Content ./kfold_output/feature_counts.txt

# Xem summary
Get-Content ./kfold_output/kfold_results/kfold_summary.csv

# Hoặc mở bằng Excel/notepad
notepad ./kfold_output/kfold_results/kfold_summary.csv
```

