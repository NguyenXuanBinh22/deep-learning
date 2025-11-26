import argparse
import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def harmonize_sample_id(sample_id: str) -> str:
    """
    Convert TCGA sample IDs like TCGA.3C.AAAU.01 -> TCGA-3C-AAAU
    to match clinical submitter_ids that usually stop at the third field.
    """
    sid = sample_id.replace(".", "-")
    parts = sid.split("-")
    return "-".join(parts[:3])


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean()
    sigma = df.std(ddof=0)
    sigma[sigma == 0] = 1.0
    return (df - mu) / sigma


def load_block(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.T
    df.index = df.index.map(harmonize_sample_id)
    return df


def build_onehot(clin: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, list]:
    labels = clin[label_col].astype(str)
    uniq = sorted(labels.dropna().unique())
    if len(uniq) < 2:
        raise ValueError(f"Label column '{label_col}' must have at least two classes, found: {uniq}")
    onehot = pd.get_dummies(labels)
    onehot = onehot[uniq]  # enforce deterministic column order
    onehot.index = clin["sample_id"]
    return onehot, uniq


def prepare_dataset(args: argparse.Namespace) -> None:
    gene_df = load_block(args.mrna_path)
    methyl_df = load_block(args.methyl_path)
    mirna_df = load_block(args.mirna_path)

    if args.top_gene:
        gene_df = gene_df.iloc[:, :args.top_gene]
    if args.top_cpg:
        methyl_df = methyl_df.iloc[:, :args.top_cpg]
    if args.top_mirna:
        mirna_df = mirna_df.iloc[:, :args.top_mirna]

    # align samples across omics, preserve original order from mRNA header
    sample_order = gene_df.index.tolist()
    if not (set(sample_order) == set(methyl_df.index) == set(mirna_df.index)):
        raise ValueError("Sample ID sets differ across omics files.")
    methyl_df = methyl_df.loc[sample_order]
    mirna_df = mirna_df.loc[sample_order]

    if args.zscore:
        gene_df = zscore(gene_df)
        methyl_df = zscore(methyl_df)
        mirna_df = zscore(mirna_df)

    num_gene = gene_df.shape[1]
    num_cpg = methyl_df.shape[1]
    num_mirna = mirna_df.shape[1]

    X = pd.concat([gene_df, methyl_df, mirna_df], axis=1)

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
        Y = Y[[str(c) for c in classes]]  # enforce deterministic ordering
        X = X.loc[sample_order]
    else:
        clin = pd.read_csv(args.clinical_path, sep="\t")
        if args.label_column is None:
            raise ValueError("label_column is required when using clinical labels.")
        if args.label_column not in clin.columns:
            raise ValueError(f"Label column '{args.label_column}' not found in clinical file.")

        # attach sample_id column aligned to omics
        clin = clin.copy()
        if "cases.submitter_id" in clin.columns:
            clin["sample_id"] = clin["cases.submitter_id"].map(harmonize_sample_id)
        elif "sample_id" in clin.columns:
            clin["sample_id"] = clin["sample_id"].map(harmonize_sample_id)
        else:
            raise ValueError("clinical.tsv must contain 'cases.submitter_id' or 'sample_id'.")

        clin = clin[clin["sample_id"].isin(sample_order)]
        onehot_y, classes = build_onehot(clin.set_index("sample_id"), args.label_column)

        # keep only samples that have labels, preserving original order
        kept = [s for s in sample_order if s in onehot_y.index]
        if not kept:
            raise ValueError("No overlapping samples between omics and labeled clinical data.")
        X = X.loc[kept]
        Y = onehot_y.loc[kept]

    stratify_labels = Y.values.argmax(axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.seed, stratify=stratify_labels
    )

    os.makedirs(args.output_dir, exist_ok=True)
    train_x_path = os.path.join(args.output_dir, "train_X.csv")
    test_x_path = os.path.join(args.output_dir, "test_X.csv")
    train_y_path = os.path.join(args.output_dir, "train_Y.csv")
    test_y_path = os.path.join(args.output_dir, "test_Y.csv")

    X_train.to_csv(train_x_path, index=False)
    X_test.to_csv(test_x_path, index=False)
    Y_train.to_csv(train_y_path, index=False)
    Y_test.to_csv(test_y_path, index=False)

    meta_path = os.path.join(args.output_dir, "feature_counts.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"num_gene={num_gene}\n")
        f.write(f"num_cpg={num_cpg}\n")
        f.write(f"num_mirna={num_mirna}\n")
        f.write(f"classes={classes}\n")

    print(f"Saved train/test splits to {args.output_dir}")
    print(f"num_gene={num_gene}, num_cpg={num_cpg}, num_mirna={num_mirna}")
    print(f"classes={classes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/test CSVs for moBRCA-net from multi-omics tables and clinical labels."
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
    parser.add_argument("--output-dir", default=".", help="Directory to write train/test CSVs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size (fraction)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--zscore", action="store_true", help="Apply per-omics z-score normalization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args)
