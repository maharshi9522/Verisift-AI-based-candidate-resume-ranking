"""
Verisift Stratified Random Pair Split Generator
Generates a mathematically rigorous 80/10/10 stratified train/val/test split
on the held-out resume-job pair level (7,987 cleaned pairs).
Saves splits and metadata to artifacts/accuracy_experiments/splits/.
"""

import os
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_stratified_pair_splits(
    input_csv: Path,
    output_dir: Path,
    seed: int = 42
):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading cleaned dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    
    # Verify label counts
    label_counts = df["label"].value_counts().to_dict()
    print("Class distribution:", label_counts)
    
    # Step 1: Split into 80% train, 20% temp (val + test)
    df_train, df_temp = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df["label"]
    )
    
    # Step 2: Split temp equally into 10% val and 10% test
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=seed,
        stratify=df_temp["label"]
    )
    
    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    print(f"Split sizes: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    assert len(df_train) + len(df_val) + len(df_test) == total_rows
    
    # Create unique pair identifiers: sha256(r_hash + '::' + j_hash)
    def pair_id(row):
        return f"{row['r_hash']}::{row['j_hash']}"
        
    train_pairs = set(df_train.apply(pair_id, axis=1))
    val_pairs = set(df_val.apply(pair_id, axis=1))
    test_pairs = set(df_test.apply(pair_id, axis=1))
    
    # Assert zero pair overlap
    overlap_train_val = train_pairs.intersection(val_pairs)
    overlap_train_test = train_pairs.intersection(test_pairs)
    overlap_val_test = val_pairs.intersection(test_pairs)
    
    assert len(overlap_train_val) == 0, f"Leakage: {len(overlap_train_val)} pairs overlap between train and val"
    assert len(overlap_train_test) == 0, f"Leakage: {len(overlap_train_test)} pairs overlap between train and test"
    assert len(overlap_val_test) == 0, f"Leakage: {len(overlap_val_test)} pairs overlap between val and test"
    print("Verification PASSED: Zero pair overlap across Train, Val, and Test splits.")
    
    # Save splits
    train_path = output_dir / "pair_held_out_train.csv"
    val_path = output_dir / "pair_held_out_val.csv"
    test_path = output_dir / "pair_held_out_test.csv"
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"Saved splits to {output_dir}")
    
    # Compute metadata
    summary = {
        "split_protocol": "stratified_random_pair_held_out",
        "description": "Stratified random split where distinct resume-job pairing interactions are held out.",
        "random_seed": seed,
        "total_pairs": total_rows,
        "splits": {
            "train": {
                "rows": len(df_train),
                "ratio": round(len(df_train) / total_rows, 4),
                "unique_resumes": int(df_train["r_hash"].nunique()),
                "unique_jds": int(df_train["j_hash"].nunique()),
                "class_distribution": df_train["label"].value_counts().to_dict(),
                "class_proportions": {k: round(v / len(df_train), 4) for k, v in df_train["label"].value_counts().items()}
            },
            "val": {
                "rows": len(df_val),
                "ratio": round(len(df_val) / total_rows, 4),
                "unique_resumes": int(df_val["r_hash"].nunique()),
                "unique_jds": int(df_val["j_hash"].nunique()),
                "class_distribution": df_val["label"].value_counts().to_dict(),
                "class_proportions": {k: round(v / len(df_val), 4) for k, v in df_val["label"].value_counts().items()}
            },
            "test": {
                "rows": len(df_test),
                "ratio": round(len(df_test) / total_rows, 4),
                "unique_resumes": int(df_test["r_hash"].nunique()),
                "unique_jds": int(df_test["j_hash"].nunique()),
                "class_distribution": df_test["label"].value_counts().to_dict(),
                "class_proportions": {k: round(v / len(df_test), 4) for k, v in df_test["label"].value_counts().items()}
            }
        },
        "leakage_verification": {
            "pair_overlap_train_val": len(overlap_train_val),
            "pair_overlap_train_test": len(overlap_train_test),
            "pair_overlap_val_test": len(overlap_val_test),
            "status": "ZERO_PAIR_OVERLAP_CONFIRMED"
        }
    }
    
    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    return summary

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    input_file = project_root / "artifacts" / "dataset_audit" / "splits" / "verisift_cleaned_all.csv"
    out_dir = project_root / "artifacts" / "accuracy_experiments" / "splits"
    create_stratified_pair_splits(input_file, out_dir, seed=42)
