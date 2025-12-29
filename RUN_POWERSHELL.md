# Hướng dẫn chạy Script PowerShell

## Các lỗi thường gặp và cách khắc phục

### 1. Lỗi Execution Policy

**Lỗi:**
```
.\run_kfold_cv.ps1 : File cannot be loaded because running scripts is disabled on this system.
```

**Giải pháp:**

**Cách 1: Cho phép script chạy (Khuyến nghị cho Current User)**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Cách 2: Bypass cho session hiện tại**
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

**Cách 3: Chạy trực tiếp với bypass**
```powershell
powershell -ExecutionPolicy Bypass -File .\run_kfold_cv.ps1
```

### 2. Lỗi "Python not found"

**Lỗi:**
```
❌ Error: Python not found. Please install Python or add it to PATH.
```

**Giải pháp:**
- Đảm bảo Python đã được cài đặt
- Kiểm tra Python có trong PATH:
  ```powershell
  python --version
  ```
- Nếu không có, thêm Python vào PATH hoặc dùng đường dẫn đầy đủ:
  ```powershell
  C:\Users\YourName\AppData\Local\Programs\Python\Python310\python.exe ...
  ```

### 3. Lỗi "file not found"

**Lỗi:**
```
❌ Error: prepare_kfold_data.py not found. Please run this script from the deep-learning-omics directory.
```

**Giải pháp:**
- Đảm bảo bạn đang ở đúng thư mục:
  ```powershell
  cd deep-learning-omics
  Get-Location  # Kiểm tra đường dẫn hiện tại
  ```

### 4. Lỗi với backtick (`) trong PowerShell

**Vấn đề:** Backtick (`) dùng để tiếp tục dòng trong PowerShell. Nếu copy-paste từ nơi khác, ký tự có thể bị thay đổi.

**Giải pháp:**
- Đảm bảo dùng backtick (`, không phải dấu nháy đơn ')
- Hoặc viết tất cả trên 1 dòng (nhưng khó đọc)

## Cách chạy đúng

### Bước 1: Mở PowerShell
- Nhấn `Win + X` và chọn "Windows PowerShell" hoặc "Terminal"
- Hoặc nhấn `Win + R`, gõ `powershell`, nhấn Enter

### Bước 2: Di chuyển đến thư mục dự án
```powershell
cd D:\semester2025.1\deep_learning\project\source_code_paper2023\deep-learning-omics
```

### Bước 3: Chạy script

**Cho model có Contrastive Learning:**
```powershell
.\run_kfold_cv.ps1
```

**Cho Baseline model:**
```powershell
.\run_kfold_cv_baseline.ps1
```

### Nếu vẫn gặp lỗi Execution Policy:

```powershell
# Cho phép script chạy (chỉ cho user hiện tại)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Sau đó chạy lại script
.\run_kfold_cv.ps1
```

## Chạy từng bước thủ công (nếu script không chạy được)

### Bước 1: Chuẩn bị dữ liệu
```powershell
python prepare_kfold_data.py --label-path data/54814634_BRCA_label_num.csv --label-column Label --zscore --output-dir ./kfold_output --k-folds 5 --top-gene 1000 --top-cpg 1000 --top-mirna 100 --seed 42
```

### Bước 2: Chạy k-fold CV
```powershell
$env:EPOCHS=50
$env:BATCH_SIZE=64
$env:LR=1e-2
python run_kfold.py --base-dir ./kfold_output --k-folds 5 --epochs 50 --batch-size 64 --lr 1e-2
```

## Kiểm tra môi trường

Chạy các lệnh sau để kiểm tra:

```powershell
# Kiểm tra Python
python --version

# Kiểm tra thư mục hiện tại
Get-Location

# Kiểm tra các file Python có tồn tại
Test-Path prepare_kfold_data.py
Test-Path run_kfold.py
Test-Path moBRCA-net.py

# Kiểm tra Execution Policy
Get-ExecutionPolicy
```

## Gợi ý

Nếu vẫn gặp vấn đề, bạn có thể:
1. Chạy từng lệnh Python trực tiếp (không qua script .ps1)
2. Dùng Git Bash hoặc WSL nếu có (chạy script .sh)
3. Chạy trong IDE như VS Code với PowerShell terminal

