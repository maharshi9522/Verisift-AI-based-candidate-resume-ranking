"""
Strict Group-Held-Out (Unseen Job Descriptions) Benchmark
Trains the winning DenseCrossNet architecture strictly on jd_held_out_train.csv
(280 JDs) and evaluates on jd_held_out_test.csv (71 zero-shot unseen JDs).
Guarantees 0 JD overlap and 0 pair overlap for true zero-shot transfer measurement.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_accuracy_experiments import (
    load_cached_embeddings, build_feature_matrix, DenseCrossFeatureNet,
    train_torch_model, evaluate_on_split, BaselinePairwiseMLP, DeepResMLP
)
from scripts.evaluate_accuracy_experiments import compute_job_ndcg

def run_unseen_jd_benchmark():
    cache_path = project_root / "artifacts" / "accuracy_experiments" / "embeddings_cache" / "minilm_512_embeddings.npz"
    r_dict, j_dict = load_cached_embeddings(cache_path)
    
    train_df = pd.read_csv(project_root / "artifacts" / "dataset_audit" / "splits" / "jd_held_out_train.csv")
    test_df = pd.read_csv(project_root / "artifacts" / "dataset_audit" / "splits" / "jd_held_out_test.csv")
    
    print(f"Loaded jd_held_out_train: {len(train_df)} rows, {train_df['j_hash'].nunique()} unique JDs")
    print(f"Loaded jd_held_out_test:  {len(test_df)} rows, {test_df['j_hash'].nunique()} unique JDs")
    
    # Verify zero JD overlap
    train_jds = set(train_df["j_hash"])
    test_jds = set(test_df["j_hash"])
    overlap = train_jds.intersection(test_jds)
    assert len(overlap) == 0, f"Error: JD overlap detected: {len(overlap)}"
    print("Verification PASSED: Exactly 0 JD overlap between train and test.")
    
    # Stratified validation split by label from training pool
    train_sub, val_sub = train_test_split(train_df, test_size=0.10, random_state=42, stratify=train_df["label"])
    
    X_train, y_train = build_feature_matrix(train_sub, r_dict, j_dict, "rich")
    X_val, y_val = build_feature_matrix(val_sub, r_dict, j_dict, "rich")
    X_test, y_test = build_feature_matrix(test_df, r_dict, j_dict, "rich")
    
    print("\nTraining DenseCrossFeatureNet on strict unseen-job training set...")
    model = DenseCrossFeatureNet(input_dim=1537, emb_dim=384)
    model, res = train_torch_model(model, X_train, y_train, X_val, y_val, epochs=40, batch_size=64, lr=8e-4)
    
    metrics = evaluate_on_split(model, X_test, y_test)
    ranking = compute_job_ndcg(test_df, metrics["predicted_scores"])
    metrics["ranking"] = ranking
    
    print("\n" + "="*70)
    print("STRICT UNSEEN JOB DESCRIPTIONS TEST SET (71 ZERO-SHOT UNSEEN JDs, 1759 ROWS)")
    print("="*70)
    print(f"Accuracy:            {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro-F1:            {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:         {metrics['weighted_f1']:.4f}")
    print("Per-class performance:")
    for cls, vals in metrics["per_class"].items():
        print(f"  {cls:15s} P={vals['precision']:.4f}  R={vals['recall']:.4f}  F1={vals['f1']:.4f}")
    print("Confusion Matrix:\n", np.array(metrics["confusion_matrix"]))
    print(f"Ranking nDCG@5:      {ranking['ndcg@5']:.4f}")
    print(f"Ranking nDCG@10:     {ranking['ndcg@10']:.4f}")
    print(f"Pairwise Concordance:{ranking['pairwise_concordance'] * 100:.2f}%")
    
    # Save metrics
    out_dir = project_root / "artifacts" / "accuracy_experiments" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_metrics = {k: v for k, v in metrics.items() if k not in ["predicted_scores", "probabilities"]}
    with open(out_dir / "strict_unseen_jd_test_metrics.json", "w") as f:
        json.dump(clean_metrics, f, indent=2)
    print(f"\nSaved strict unseen JD metrics to {out_dir / 'strict_unseen_jd_test_metrics.json'}")
    return metrics

if __name__ == "__main__":
    run_unseen_jd_benchmark()
