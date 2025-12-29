#!/bin/bash

# Script để chạy Stratified K-Fold Cross-Validation cho moBRCA-net
# Sử dụng: bash run_kfold_cv.sh

# Cấu hình
K_FOLDS=5
OUTPUT_DIR="./kfold_output"
EPOCHS=50
BATCH_SIZE=64
LR=1e-2

# Bước 1: Chuẩn bị dữ liệu k-fold
echo "Step 1: Preparing k-fold data splits..."
python prepare_kfold_data.py \
    --label-path data/54814634_BRCA_label_num.csv \
    --label-column Label \
    --zscore \
    --output-dir "$OUTPUT_DIR" \
    --k-folds $K_FOLDS \
    --top-gene 1000 --top-cpg 1000 --top-mirna 100 \
    --seed 42

# Bước 2: Chạy k-fold cross-validation
echo ""
echo "Step 2: Running k-fold cross-validation..."
export EPOCHS=$EPOCHS
export BATCH_SIZE=$BATCH_SIZE
export LR=$LR

python run_kfold.py \
    --base-dir "$OUTPUT_DIR" \
    --k-folds $K_FOLDS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR

echo ""
echo "✅ K-fold cross-validation completed!"
echo "📊 Results are saved in: $OUTPUT_DIR/kfold_results/"

