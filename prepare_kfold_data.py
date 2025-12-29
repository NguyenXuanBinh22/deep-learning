"""
Script để chuẩn bị dữ liệu cho Stratified K-Fold Cross-Validation
Tạo k folds từ multi-omics data với stratified splitting
"""
import argparse
import os
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def harmonize_sample_id(sample_id: str) -> str:
    """
    Convert TCGA sample IDs like TCGA.3C.AAAU.01 -> TCGA-3C-AAAU
    to match clinical submitter_ids that usually stop at the third field.
    """
    sid = sample_id.replace(".", "-")
    parts = sid.split("-")
    return "-".join(parts[:3])


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Apply z-score normalization per feature."""
    mu = df.mean()
    sigma = df.std(ddof=0)
    sigma[sigma == 0] = 1.0
    return (df - mu) / sigma


def load_block(path: str) -> pd.DataFrame:
    """Load omics block and transpose (rows=samples, cols=features)."""
    df = pd.read_csv(path, index_col=0)
    df = df.T
    df.index = df.index.map(harmonize_sample_id)
    return df


def build_onehot(clin: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, list]:
    """Build one-hot encoded labels."""
    labels = clin[label_col].astype(str)
    uniq = sorted(labels.dropna().unique())
    if len(uniq) < 2:
        raise ValueError(f"Label column '{label_col}' must have at least two classes, found: {uniq}")
    onehot = pd.get_dummies(labels)
    onehot = onehot[uniq]  # enforce deterministic column order
    onehot.index = clin["sample_id"]
    return onehot, uniq


def prepare_kfold_dataset(args: argparse.Namespace) -> None:
    """
    Chuẩn bị dữ liệu và chia thành k folds với stratified splitting.
    """
    # Load omics data
    gene_df = load_block(args.mrna_path)
    methyl_df = load_block(args.methyl_path)
    mirna_df = load_block(args.mirna_path)

    # Limit features if specified
    if args.top_gene:
        gene_df = gene_df.iloc[:, :args.top_gene]
    if args.top_cpg:
        methyl_df = methyl_df.iloc[:, :args.top_cpg]
    if args.top_mirna:
        mirna_df = mirna_df.iloc[:, :args.top_mirna]

    # Align samples across omics
    sample_order = gene_df.index.tolist()
    if not (set(sample_order) == set(methyl_df.index) == set(mirna_df.index)):
        raise ValueError("Sample ID sets differ across omics files.")
    methyl_df = methyl_df.loc[sample_order]
    mirna_df = mirna_df.loc[sample_order]

    # Apply z-score normalization if requested
    if args.zscore:
        gene_df = zscore(gene_df)
        methyl_df = zscore(methyl_df)
        mirna_df = zscore(mirna_df)

    num_gene = gene_df.shape[1]
    num_cpg = methyl_df.shape[1]
    num_mirna = mirna_df.shape[1]

    # Concatenate omics
    X = pd.concat([gene_df, methyl_df, mirna_df], axis=1)

    # Load labels
    if args.label_path:
        labels_df = pd.read_csv(args.label_path)
        label_col = args.label_column or labels_df.columns[0]
        if label_col not in labels_df.columns:
            raise ValueError(f"Label column '{label_col}' not found in label file.")
        labels = labels_df[label_col].values
        if len(labels) != len(sample_order):
            raise ValueError(
                f"Label count {len(labels)} does not match sample count {len(sample_order)}."
            )
        classes = sorted(pd.unique(labels))
        labels_series = pd.Series(labels, index=sample_order, name="label")
        Y = pd.get_dummies(labels_series.astype(str))
        Y = Y[[str(c) for c in classes]]
        X = X.loc[sample_order]
    else:
        clin = pd.read_csv(args.clinical_path, sep="\t")
        if args.label_column is None:
            raise ValueError("label_column is required when using clinical labels.")
        if args.label_column not in clin.columns:
            raise ValueError(f"Label column '{args.label_column}' not found in clinical file.")

        clin = clin.copy()
        if "cases.submitter_id" in clin.columns:
            clin["sample_id"] = clin["cases.submitter_id"].map(harmonize_sample_id)
        elif "sample_id" in clin.columns:
            clin["sample_id"] = clin["sample_id"].map(harmonize_sample_id)
        else:
            raise ValueError("clinical.tsv must contain 'cases.submitter_id' or 'sample_id'.")

        clin = clin[clin["sample_id"].isin(sample_order)]
        onehot_y, classes = build_onehot(clin.set_index("sample_id"), args.label_column)

        kept = [s for s in sample_order if s in onehot_y.index]
        if not kept:
            raise ValueError("No overlapping samples between omics and labeled clinical data.")
        X = X.loc[kept]
        Y = onehot_y.loc[kept]

    # Convert labels to class indices for stratification
    stratify_labels = Y.values.argmax(axis=1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save feature counts
    meta_path = os.path.join(args.output_dir, "feature_counts.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"num_gene={num_gene}\n")
        f.write(f"num_cpg={num_cpg}\n")
        f.write(f"num_mirna={num_mirna}\n")
        f.write(f"classes={classes}\n")

    print(f"Total samples: {len(X)}")
    print(f"num_gene={num_gene}, num_cpg={num_cpg}, num_mirna={num_mirna}")
    print(f"classes={classes}")
    print(f"Class distribution: {pd.Series(stratify_labels).value_counts().sort_index().to_dict()}\n")

    # Create StratifiedKFold
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    # Create folds directory
    folds_dir = os.path.join(args.output_dir, "folds")
    os.makedirs(folds_dir, exist_ok=True)

    print(f"Creating {args.k_folds} stratified folds...")
    
    # Split into k folds
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, stratify_labels), 1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        Y_train = Y.iloc[train_idx]
        Y_test = Y.iloc[test_idx]

        fold_dir = os.path.join(folds_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # Save train/test splits for this fold
        train_x_path = os.path.join(fold_dir, "train_X.csv")
        test_x_path = os.path.join(fold_dir, "test_X.csv")
        train_y_path = os.path.join(fold_dir, "train_Y.csv")
        test_y_path = os.path.join(fold_dir, "test_Y.csv")

        X_train.to_csv(train_x_path, index=False)
        X_test.to_csv(test_x_path, index=False)
        Y_train.to_csv(train_y_path, index=False)
        Y_test.to_csv(test_y_path, index=False)

        # Print fold statistics
        train_labels = Y_train.values.argmax(axis=1)
        test_labels = Y_test.values.argmax(axis=1)
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(X_train)} samples - {pd.Series(train_labels).value_counts().sort_index().to_dict()}")
        print(f"  Test:  {len(X_test)} samples  - {pd.Series(test_labels).value_counts().sort_index().to_dict()}")

    print(f"\n✅ Saved {args.k_folds} folds to {folds_dir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare stratified k-fold cross-validation splits for moBRCA-net."
    )
    parser.add_argument("--mrna-path", default="data/BRCA_mRNA_top.csv", help="Path to mRNA features csv")
    parser.add_argument("--methyl-path", default="data/BRCA_Methy_top.csv", help="Path to methylation features csv")
    parser.add_argument("--mirna-path", default="data/BRCA_miRNA_top.csv", help="Path to miRNA features csv")
    parser.add_argument("--clinical-path", default="clinical.tsv", help="Clinical TSV with subtype labels")
    parser.add_argument("--label-column", help="Column in clinical or label file that holds subtype labels")
    parser.add_argument("--label-path", help="Optional CSV with labels aligned to sample order in feature files")
    parser.add_argument("--top-gene", type=int, help="Use only the first N gene features")
    parser.add_argument("--top-cpg", type=int, help="Use only the first N methylation features")
    parser.add_argument("--top-mirna", type=int, help="Use only the first N miRNA features")
    parser.add_argument("--output-dir", default=".", help="Directory to write fold splits")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--zscore", action="store_true", help="Apply per-omics z-score normalization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_kfold_dataset(args)

