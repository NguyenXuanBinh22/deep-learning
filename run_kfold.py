"""
Script để chạy Stratified K-Fold Cross-Validation
Train model trên từng fold và tổng hợp kết quả
"""
import os
import sys
import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

# Import training function từ moBRCA-net (file có dấu gạch ngang nên dùng importlib)
# Tìm đường dẫn đến moBRCA-net.py dựa trên thư mục của script này
script_dir = os.path.dirname(os.path.abspath(__file__))
mobrca_net_path = os.path.join(script_dir, "moBRCA-net.py")
spec = importlib.util.spec_from_file_location("mobrca_net", mobrca_net_path)
mobrca_net = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mobrca_net)
train_and_eval = mobrca_net.train_and_eval
MultiOmicsDataset = mobrca_net.MultiOmicsDataset


def load_fold_data(fold_dir, n_gene, n_methyl, n_mirna):
    """Load data cho một fold."""
    train_x_path = os.path.join(fold_dir, "train_X.csv")
    test_x_path = os.path.join(fold_dir, "test_X.csv")
    train_y_path = os.path.join(fold_dir, "train_Y.csv")
    test_y_path = os.path.join(fold_dir, "test_Y.csv")

    x_train = pd.read_csv(train_x_path, dtype=np.float32)
    x_test = pd.read_csv(test_x_path, dtype=np.float32)
    y_train = pd.read_csv(train_y_path, dtype=np.float32).values
    y_test = pd.read_csv(test_y_path, dtype=np.float32).values

    # Tách theo omics
    x_gene_train = x_train.iloc[:, :n_gene].values
    x_gene_test = x_test.iloc[:, :n_gene].values

    x_methyl_train = x_train.iloc[:, n_gene:n_gene + n_methyl].values
    x_methyl_test = x_test.iloc[:, n_gene:n_gene + n_methyl].values

    x_mirna_train = x_train.iloc[:, n_gene + n_methyl:].values
    x_mirna_test = x_test.iloc[:, n_gene + n_methyl:].values

    train_ds = MultiOmicsDataset(x_gene_train, x_methyl_train, x_mirna_train, y_train)
    test_ds = MultiOmicsDataset(x_gene_test, x_methyl_test, x_mirna_test, y_test)

    return train_ds, test_ds, y_test


def read_feature_counts(feature_counts_path):
    """Đọc số lượng features từ file."""
    counts = {}
    with open(feature_counts_path, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            if key in ['num_gene', 'num_cpg', 'num_mirna']:
                counts[key] = int(value)
            elif key == 'classes':
                # Parse classes: [0, 1, 2, 3, 4]
                value = value.strip('[]')
                counts[key] = [int(x.strip()) for x in value.split(',')]
    return counts


def run_kfold_cv(args):
    """
    Chạy k-fold cross-validation.
    """
    base_dir = args.base_dir
    k_folds = args.k_folds
    
    # Đọc feature counts
    feature_counts_path = os.path.join(base_dir, "feature_counts.txt")
    if not os.path.exists(feature_counts_path):
        raise FileNotFoundError(f"Feature counts file not found: {feature_counts_path}")
    
    counts = read_feature_counts(feature_counts_path)
    n_gene = counts['num_gene']
    n_methyl = counts['num_cpg']
    n_mirna = counts['num_mirna']
    n_classes = len(counts['classes'])

    print("=" * 80)
    print("STRATIFIED K-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Number of folds: {k_folds}")
    print(f"Features: gene={n_gene}, cpg={n_methyl}, mirna={n_mirna}")
    print(f"Classes: {n_classes}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print("=" * 80 + "\n")

    # Tạo thư mục kết quả tổng hợp
    results_dir = os.path.join(base_dir, "kfold_results")
    os.makedirs(results_dir, exist_ok=True)

    # Lưu trữ metrics cho tất cả folds
    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'fold': []
    }
    
    # Lưu trữ predictions và labels cho tất cả folds
    all_predictions = []
    all_labels = []

    # Chạy training trên từng fold
    for fold_idx in range(1, k_folds + 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}/{k_folds}")
        print(f"{'='*80}\n")

        fold_dir = os.path.join(base_dir, "folds", f"fold_{fold_idx}")
        if not os.path.exists(fold_dir):
            print(f"⚠️  Warning: Fold {fold_idx} directory not found: {fold_dir}")
            continue

        # Load data cho fold này
        train_ds, test_ds, y_test_true = load_fold_data(fold_dir, n_gene, n_methyl, n_mirna)

        # Thư mục kết quả cho fold này
        fold_results_dir = os.path.join(results_dir, f"fold_{fold_idx}")
        os.makedirs(fold_results_dir, exist_ok=True)

        # Train và evaluate
        print(f"Training on fold {fold_idx}...")
        train_and_eval(
            train_ds=train_ds,
            test_ds=test_ds,
            n_gene=n_gene,
            n_methyl=n_methyl,
            n_mirna=n_mirna,
            n_classes=n_classes,
            res_dir=fold_results_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            dropout=args.dropout,
            weight_decay=args.weight_decay
        )

        # Đọc kết quả từ fold này
        pred_path = os.path.join(fold_results_dir, "prediction.csv")
        label_path = os.path.join(fold_results_dir, "label.csv")

        if os.path.exists(pred_path) and os.path.exists(label_path):
            y_pred = np.loadtxt(pred_path, delimiter=',', dtype=int)
            y_true = np.loadtxt(label_path, delimiter=',', dtype=int)

            # Tính metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            fold_metrics['accuracy'].append(acc)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)
            fold_metrics['f1'].append(f1)
            fold_metrics['fold'].append(fold_idx)

            # Lưu predictions và labels để tổng hợp
            all_predictions.extend(y_pred.tolist())
            all_labels.extend(y_true.tolist())

            print(f"\n📊 Fold {fold_idx} Results:")
            print(f"   Accuracy:  {acc:.4f}")
            print(f"   Precision: {prec:.4f}")
            print(f"   Recall:    {rec:.4f}")
            print(f"   F1-Score:  {f1:.4f}")

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            print(f"\n   Confusion Matrix:")
            print(f"   {cm}")
        else:
            print(f"⚠️  Warning: Results not found for fold {fold_idx}")

    # Tổng hợp kết quả
    print(f"\n{'='*80}")
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}\n")

    if len(fold_metrics['accuracy']) > 0:
        # Tính mean và std
        summary = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Mean': [
                np.mean(fold_metrics['accuracy']),
                np.mean(fold_metrics['precision']),
                np.mean(fold_metrics['recall']),
                np.mean(fold_metrics['f1'])
            ],
            'Std': [
                np.std(fold_metrics['accuracy']),
                np.std(fold_metrics['precision']),
                np.std(fold_metrics['recall']),
                np.std(fold_metrics['f1'])
            ],
            'Min': [
                np.min(fold_metrics['accuracy']),
                np.min(fold_metrics['precision']),
                np.min(fold_metrics['recall']),
                np.min(fold_metrics['f1'])
            ],
            'Max': [
                np.max(fold_metrics['accuracy']),
                np.max(fold_metrics['precision']),
                np.max(fold_metrics['recall']),
                np.max(fold_metrics['f1'])
            ]
        }

        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))

        # In chi tiết từng fold
        print(f"\n📋 Per-Fold Results:")
        for i in range(len(fold_metrics['fold'])):
            print(f"   Fold {fold_metrics['fold'][i]}: "
                  f"Acc={fold_metrics['accuracy'][i]:.4f}, "
                  f"Prec={fold_metrics['precision'][i]:.4f}, "
                  f"Rec={fold_metrics['recall'][i]:.4f}, "
                  f"F1={fold_metrics['f1'][i]:.4f}")

        # Lưu summary vào file
        summary_path = os.path.join(results_dir, "kfold_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ Summary saved to: {summary_path}")

        # Lưu per-fold metrics
        per_fold_df = pd.DataFrame({
            'Fold': fold_metrics['fold'],
            'Accuracy': fold_metrics['accuracy'],
            'Precision': fold_metrics['precision'],
            'Recall': fold_metrics['recall'],
            'F1-Score': fold_metrics['f1']
        })
        per_fold_path = os.path.join(results_dir, "per_fold_metrics.csv")
        per_fold_df.to_csv(per_fold_path, index=False)
        print(f"✅ Per-fold metrics saved to: {per_fold_path}")

        # Lưu tất cả predictions và labels
        all_pred_path = os.path.join(results_dir, "all_predictions.csv")
        all_label_path = os.path.join(results_dir, "all_labels.csv")
        np.savetxt(all_pred_path, all_predictions, fmt="%d", delimiter=",")
        np.savetxt(all_label_path, all_labels, fmt="%d", delimiter=",")
        print(f"✅ All predictions saved to: {all_pred_path}")
        print(f"✅ All labels saved to: {all_label_path}")

        # Tính overall metrics trên tất cả predictions
        overall_acc = accuracy_score(all_labels, all_predictions)
        overall_prec = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        overall_rec = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        overall_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

        print(f"\n📊 Overall Metrics (pooled across all folds):")
        print(f"   Accuracy:  {overall_acc:.4f}")
        print(f"   Precision: {overall_prec:.4f}")
        print(f"   Recall:    {overall_rec:.4f}")
        print(f"   F1-Score:  {overall_f1:.4f}")

        # Overall confusion matrix
        overall_cm = confusion_matrix(all_labels, all_predictions)
        print(f"\n   Overall Confusion Matrix:")
        print(f"   {overall_cm}")

        # Lưu overall confusion matrix
        cm_path = os.path.join(results_dir, "overall_confusion_matrix.csv")
        np.savetxt(cm_path, overall_cm, fmt="%d", delimiter=",")
        print(f"✅ Overall confusion matrix saved to: {cm_path}")

    else:
        print("❌ No results found!")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run k-fold cross-validation for moBRCA-net"
    )
    parser.add_argument("--base-dir", required=True, 
                       help="Base directory containing 'folds' and 'feature_counts.txt'")
    parser.add_argument("--k-folds", type=int, default=5,
                       help="Number of folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (default: from EPOCHS env or 200)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (default: from BATCH_SIZE env or 136)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (default: from LR env or 1e-2)")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate (default: 0.2)")
    parser.add_argument("--weight-decay", type=float, default=None,
                       help="Weight decay (default: from WEIGHT_DECAY env or 1e-4)")
    
    args = parser.parse_args()
    
    # Set defaults from environment if not provided
    if args.epochs is None:
        args.epochs = int(os.getenv("EPOCHS", "200"))
    if args.batch_size is None:
        args.batch_size = int(os.getenv("BATCH_SIZE", "136"))
    if args.lr is None:
        args.lr = float(os.getenv("LR", "1e-2"))
    if args.weight_decay is None:
        args.weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    
    return args


if __name__ == "__main__":
    args = parse_args()
    run_kfold_cv(args)

