# moBRCA-net: a Breast Cancer Subtype Classification Framework Based on Multi-Omics Attention Neural Networks
moBRCA-net is an omics-level attention-based breast cancer subtype classification framework that uses multi-omics datasets. Dataset integration was performed based on feature-selection modules that consider the biological relationship between the omics datasets (gene expression, DNA methylation, and microRNA expression). Moreover, for omics-level feature importance learning, a self-attention module was applied for each omics feature, and each feature was then transformed to the new representation incorporating its relative importance for the classification task. The representation of each omics dataset was concatenated and delivered to the fully connected layers to predict the breast cancer subtype of each patient.

![Figure](https://github.com/cbi-bioinfo/moBRCA-net/blob/main/fig1_v7.png?raw=true)

## Requirements
* Python >= 3.9
* PyTorch >= 2.2 (CPU OK)
* numpy >= 1.26, pandas, scikit-learn (for data prep)

## Usage
Clone the repository or download source code files and prepare breast cancer multi-omics dataset including gene expression, DNA methylation, and microRNA expression.

### Quick start (data prep + train)
1) Chuẩn bị dữ liệu thô:  
   - `data/BRCA_mRNA_top.csv`, `data/BRCA_Methy_top.csv`, `data/BRCA_miRNA_top.csv` (các hàng là feature, các cột là sample).  
   - File nhãn `data/xxx_label.csv` (cột `Label` là số hoặc tên subtype, thứ tự dòng khớp thứ tự sample trong các file omics), hoặc `clinical.tsv` có nhãn.

2) Tạo 4 file đầu vào và ghi số feature:
```
  python prepare_data.py \
    --label-path data/54814634_BRCA_label_num.csv \
    --label-column Label \
    --zscore \
    --output-dir . \
    --test-size 0.2 \
    --top-gene 1000 --top-cpg 1000 --top-mirna 100
```
Kết quả: `train_X.csv`, `test_X.csv`, `train_Y.csv`, `test_Y.csv`, `feature_counts.txt` (num_gene/num_cpg/num_mirna).

3) Cập nhật `run_moBRCA-net.sh` hoặc gọi trực tiếp:
```
# env tùy chọn để chạy nhanh hơn
$env:EPOCHS=50; $env:BATCH_SIZE=64;
python moBRCA-net.py train_X.csv train_Y.csv test_X.csv test_Y.csv 500 500 100 results
```

4) Đầu ra ở thư mục `results/`:  
   - `prediction.csv`, `label.csv`  
   - `attn_score_gene.csv`, `attn_score_methyl.csv`, `attn_score_mirna.csv`

## Conditional variational autoencoder for data augmentation used in the experiment
* cvae_generator.py

## Contact
If you have any question or problem, please send an email to **joungmin AT vt.edu**
