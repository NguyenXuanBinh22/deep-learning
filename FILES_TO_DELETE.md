# Danh sách files có thể xóa khi chạy thủ công

## 🔴 KHÔNG XÓA - Files cần thiết:

### Python scripts chính:
- ✅ `moBRCA-net.py` - Model chính với Contrastive Learning
- ✅ `moBRCA-net_baseline.py` - Model baseline
- ✅ `prepare_kfold_data.py` - Chuẩn bị dữ liệu k-fold
- ✅ `prepare_data.py` - Chuẩn bị dữ liệu train/test split
- ✅ `run_kfold.py` - Chạy k-fold CV (model có CL)
- ✅ `run_kfold_baseline.py` - Chạy k-fold CV (baseline)

### Support files:
- ✅ `contrast.py` - Module contrastive learning (cần cho moBRCA-net.py)
- ✅ `merge_and_kfold.py` - Nếu bạn muốn gộp files (tùy chọn)

### Dữ liệu:
- ✅ `data/` - Thư mục chứa dữ liệu gốc (KHÔNG XÓA!)
  - `BRCA_mRNA_top.csv`
  - `BRCA_Methy_top.csv`
  - `BRCA_miRNA_top.csv`
  - `54814634_BRCA_label_num.csv`

---

## 🟡 CÓ THỂ XÓA - Script tự động (không cần khi chạy thủ công):

### Scripts tự động:
- ❌ `run_kfold_cv.ps1` - PowerShell script tự động
- ❌ `run_kfold_cv.sh` - Bash script tự động
- ❌ `run_kfold_cv_baseline.ps1` - PowerShell script baseline
- ❌ `run_kfold_cv_baseline.sh` - Bash script baseline
- ❌ `run_moBRCA-net.sh` - Bash script cho train/test split

**Lý do:** Chỉ cần khi chạy tự động, không cần khi chạy thủ công từng lệnh Python.

---

## 🟡 CÓ THỂ XÓA - Kết quả (để chạy lại từ đầu):

### Thư mục kết quả:
- ⚠️ `kfold_output/` - Kết quả k-fold CV (có thể xóa để chạy lại)
- ⚠️ `kfold_output_merged/` - Kết quả từ merge_and_kfold (có thể xóa)
- ⚠️ `results/` - Kết quả train/test split cũ (có thể xóa)

**Lưu ý:** Nếu muốn chạy lại từ đầu, có thể xóa các thư mục này.

---

## 🟡 CÓ THỂ XÓA - Files trung gian (nếu đã có trong output):

### Files ở thư mục gốc (nếu đã được tạo trong output dir):
- ⚠️ `train_X.csv`, `train_Y.csv`, `test_X.csv`, `test_Y.csv` ở thư mục gốc
- ⚠️ `feature_counts.txt` ở thư mục gốc (nếu đã có trong output dir)

**Lưu ý:** Chỉ xóa nếu bạn đã có chúng trong `kfold_output/` hoặc thư mục output khác.

---

## 🟢 TÙY CHỌN XÓA - Files không bắt buộc:

### Optional scripts:
- ❓ `cvae_generator.py` - Chỉ cần nếu dùng data augmentation
- ❓ `merge_and_kfold.py` - Chỉ cần nếu muốn gộp files

### Documentation:
- ❓ `README.md` - Có thể xóa nếu không cần đọc
- ❓ `KFOLD_README.md` - Có thể xóa nếu không cần
- ❓ `MERGE_KFOLD_README.md` - Có thể xóa nếu không cần
- ❓ `RUN_MANUAL.md` - Có thể xóa nếu không cần

### Khác:
- ❓ `fig1_v7.png` - Hình minh họa, không cần cho code
- ❓ `LICENSE` - Giấy phép, nên giữ
- ❓ `dll2/` - Thư mục này, cần kiểm tra xem là gì

---

## 📋 TÓM TẮT NHANH:

### Nếu chạy thủ công, bạn CẦN:
```
✅ moBRCA-net.py
✅ moBRCA-net_baseline.py  
✅ prepare_kfold_data.py
✅ prepare_data.py
✅ run_kfold.py
✅ run_kfold_baseline.py
✅ contrast.py
✅ data/ (toàn bộ thư mục)
```

### Có thể XÓA để gọn:
```
❌ *.ps1 (PowerShell scripts)
❌ *.sh (Bash scripts)
❌ kfold_output/ (kết quả - có thể tạo lại)
❌ results/ (kết quả - có thể tạo lại)
❌ *.md (documentation - tùy bạn)
```

---

## 💡 LỆNH XÓA NHANH:

### Windows PowerShell (xóa scripts tự động):
```powershell
Remove-Item run_kfold_cv.ps1, run_kfold_cv.sh, run_kfold_cv_baseline.ps1, run_kfold_cv_baseline.sh, run_moBRCA-net.sh
```

### Xóa kết quả cũ (nếu muốn chạy lại):
```powershell
Remove-Item -Recurse -Force kfold_output, kfold_output_merged, results
```

### Xóa files trung gian (nếu đã có trong output):
```powershell
Remove-Item train_X.csv, train_Y.csv, test_X.csv, test_Y.csv, feature_counts.txt
```

⚠️ **CẢNH BÁO:** Chỉ xóa sau khi đã backup hoặc chắc chắn không cần nữa!

