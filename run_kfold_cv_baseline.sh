#!/bin/bash

# Script để chạy Stratified K-Fold Cross-Validation cho moBRCA-net BASELINE
# Sử dụng: bash run_kfold_cv_baseline.sh

# Cấu hình
K_FOLDS=5
OUTPUT_DIR="./kfold_output"
EPOCHS=50
BATCH_SIZE=64
LR=1e-2

# Bước 1: Chuẩn bị dữ liệu k-fold (có thể dùng chung với version có contrastive learning)
echo "Step 1: Checking k-fold data splits..."
if [ ! -d "$OUTPUT_DIR/folds" ]; then
    echo "Folds not found. Preparing k-fold data splits..."
    python prepare_kfold_data.py \
        --label-path data/54814634_BRCA_label_num.csv \
        --label-column Label \
        --zscore \
        --output-dir "$OUTPUT_DIR" \
        --k-folds $K_FOLDS \
        --top-gene 1000 --top-cpg 1000 --top-mirna 100 \
        --seed 42
else
    echo "Folds already exist. Skipping data preparation."
fi

# Bước 2: Chạy k-fold cross-validation cho BASELINE
echo ""
echo "Step 2: Running k-fold cross-validation for BASELINE model..."
export EPOCHS=$EPOCHS
export BATCH_SIZE=$BATCH_SIZE
export LR=$LR

python run_kfold_baseline.py \
    --base-dir "$OUTPUT_DIR" \
    --k-folds $K_FOLDS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR

echo ""
echo "✅ K-fold cross-validation for BASELINE completed!"
echo "📊 Results are saved in: $OUTPUT_DIR/kfold_results_baseline/"

