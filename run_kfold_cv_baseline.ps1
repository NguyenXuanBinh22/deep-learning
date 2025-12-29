# PowerShell script để chạy Stratified K-Fold Cross-Validation cho moBRCA-net BASELINE
# Sử dụng: .\run_kfold_cv_baseline.ps1
# Nếu gặp lỗi execution policy: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Kiểm tra và xử lý lỗi
$ErrorActionPreference = "Stop"

# Kiểm tra Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Python not found. Please install Python or add it to PATH." -ForegroundColor Red
    exit 1
}

# Kiểm tra đang ở đúng thư mục
if (-not (Test-Path "prepare_kfold_data.py")) {
    Write-Host "❌ Error: prepare_kfold_data.py not found. Please run this script from the deep-learning-omics directory." -ForegroundColor Red
    exit 1
}

# Cấu hình
$K_FOLDS = 5
$OUTPUT_DIR = "./kfold_output"
$EPOCHS = 50
$BATCH_SIZE = 64
$LR = 1e-2
# Số lượng features
$TOP_GENE = 1000
$TOP_CPG = 1000
$TOP_MIRNA = 100

# Bước 1: Chuẩn bị dữ liệu k-fold (có thể dùng chung với version có contrastive learning)
Write-Host "Step 1: Checking k-fold data splits..." -ForegroundColor Cyan
if (-not (Test-Path "$OUTPUT_DIR/folds")) {
    Write-Host "Folds not found. Preparing k-fold data splits..." -ForegroundColor Yellow
    try {
        python prepare_kfold_data.py `
            --label-path data/54814634_BRCA_label_num.csv `
            --label-column Label `
            --zscore `
            --output-dir $OUTPUT_DIR `
            --k-folds $K_FOLDS `
            --top-gene $TOP_GENE --top-cpg $TOP_CPG --top-mirna $TOP_MIRNA `
            --seed 42
        
        if ($LASTEXITCODE -ne 0) {
            throw "prepare_kfold_data.py failed with exit code $LASTEXITCODE"
        }
        Write-Host "✅ Data preparation completed!" -ForegroundColor Green
    } catch {
        Write-Host "❌ Error in data preparation: $_" -ForegroundColor Red
        exit 1
}
} else {
    Write-Host "Folds already exist. Skipping data preparation." -ForegroundColor Green
}

# Bước 2: Chạy k-fold cross-validation cho BASELINE
Write-Host ""
Write-Host "Step 2: Running k-fold cross-validation for BASELINE model..." -ForegroundColor Cyan
$env:EPOCHS = $EPOCHS
$env:BATCH_SIZE = $BATCH_SIZE
$env:LR = $LR

try {
    python run_kfold_baseline.py `
        --base-dir $OUTPUT_DIR `
        --k-folds $K_FOLDS `
        --epochs $EPOCHS `
        --batch-size $BATCH_SIZE `
        --lr $LR
    
    if ($LASTEXITCODE -ne 0) {
        throw "run_kfold_baseline.py failed with exit code $LASTEXITCODE"
    }
} catch {
    Write-Host "❌ Error in k-fold cross-validation: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ K-fold cross-validation for BASELINE completed!" -ForegroundColor Green
Write-Host "📊 Results are saved in: $OUTPUT_DIR/kfold_results_baseline/" -ForegroundColor Green
