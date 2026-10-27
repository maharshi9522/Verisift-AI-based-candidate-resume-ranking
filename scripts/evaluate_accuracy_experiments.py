"""
Verisift Independent Test Evaluator & Double-Split Benchmark
Evaluates the best validated model on:
1. Held-out resume-job pair test split (pair_held_out_test.csv - 799 rows)
2. Zero-shot unseen job descriptions test split (jd_held_out_test.csv - 1,759 rows)
Computes complete classification metrics, confusion matrices, and ranking concordance.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
from pathlib import Path
import json
import time
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
INV_LABEL_MAP = {0: "No Fit", 1: "Potential Fit", 2: "Good Fit"}

# Import architectures from run_accuracy_experiments
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from scripts.run_accuracy_experiments import (
    BaselinePairwiseMLP, DeepResMLP, DenseCrossFeatureNet,
    load_cached_embeddings, build_feature_matrix, evaluate_on_split
)

def compute_job_ndcg(
    df: pd.DataFrame,
    predicted_scores: np.ndarray,
    k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    """Computes mean nDCG@K across distinct job descriptions."""
    df_eval = df.copy()
    df_eval["score"] = predicted_scores
    df_eval["true_num"] = df_eval["label"].map(LABEL_MAP)
    
    ndcg_results = {f"ndcg@{k}": [] for k in k_values}
    concordant = 0
    discordant = 0
    ties = 0
    total_pairs = 0
    
    for _, group in df_eval.groupby("j_hash"):
        if len(group) < 2:
            continue
        y_true = group["true_num"].to_numpy()
        y_score = group["score"].to_numpy()
        
        # Concordance within job
        for i in range(len(y_true)):
            for j in range(i + 1, len(y_true)):
                if y_true[i] == y_true[j]:
                    continue
                total_pairs += 1
                if (y_true[i] > y_true[j] and y_score[i] > y_score[j]) or (y_true[i] < y_true[j] and y_score[i] < y_score[j]):
                    concordant += 1
                elif y_score[i] == y_score[j]:
                    ties += 1
                else:
                    discordant += 1
                    
        # nDCG calculation
        order = np.argsort(y_score)[::-1]
        sorted_true = y_true[order]
        ideal_order = np.argsort(y_true)[::-1]
        ideal_true = y_true[ideal_order]
        
        for k in k_values:
            curr_sorted = sorted_true[:k]
            curr_ideal = ideal_true[:k]
            
            discounts = np.log2(np.arange(2, len(curr_sorted) + 2))
            dcg = np.sum((2.0 ** curr_sorted - 1.0) / discounts)
            idcg = np.sum((2.0 ** curr_ideal - 1.0) / discounts)
            
            if idcg > 0:
                ndcg_results[f"ndcg@{k}"].append(dcg / idcg)
                
    summary = {k: round(float(np.mean(v)), 4) if len(v) > 0 else 0.0 for k, v in ndcg_results.items()}
    concordance_ratio = concordant / total_pairs if total_pairs > 0 else 0.0
    summary["pairwise_concordance"] = round(concordance_ratio, 4)
    summary["evaluated_jobs"] = int(df_eval["j_hash"].nunique())
    summary["total_evaluated_pairs"] = total_pairs
    return summary

def run_independent_test_evaluation():
    splits_dir = project_root / "artifacts" / "accuracy_experiments" / "splits"
    cache_dir = project_root / "artifacts" / "accuracy_experiments" / "embeddings_cache"
    checkpoints_dir = project_root / "artifacts" / "accuracy_experiments" / "checkpoints"
    metrics_dir = project_root / "artifacts" / "accuracy_experiments" / "metrics"
    
    # 1. Load best model configuration
    config_file = checkpoints_dir / "model_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing {config_file}. Run run_accuracy_experiments.py first.")
        
    with open(config_file, "r") as f:
        config = json.load(f)
        
    print(f"Loaded Best Model Configuration: {config['model_id']} ({config['architecture']})")
    print(f"Feature mode: {config['feature_mode']}, Context: {config['context_tokens']} tokens, Input dim: {config['input_dim']}")
    
    # 2. Re-instantiate architecture and load weights
    arch_name = config["architecture"]
    input_dim = config["input_dim"]
    
    if arch_name == "DeepResMLP":
        model = DeepResMLP(input_dim=input_dim, hidden_dim=512)
    elif arch_name == "BaselinePairwiseMLP":
        model = BaselinePairwiseMLP(input_dim=input_dim)
    elif arch_name == "DenseCrossFeatureNet":
        model = DenseCrossFeatureNet(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
        
    model.load_state_dict(torch.load(checkpoints_dir / "best_deep_model.pt", map_location="cpu"))
    model.eval()
    print("Model weights successfully loaded.")
    
    # 3. Load embeddings cache
    cache_file = cache_dir / f"minilm_{config['context_tokens']}_embeddings.npz"
    r_dict, j_dict = load_cached_embeddings(cache_file)
    
    # ---------------- TEST EVALUATION 1: Held-out Pair Test Set ----------------
    print("\n" + "="*70)
    print("TEST EVALUATION 1: Held-out Resume-Job Pair Split (pair_held_out_test.csv)")
    print("="*70)
    test_df = pd.read_csv(splits_dir / "pair_held_out_test.csv")
    X_test, y_test = build_feature_matrix(test_df, r_dict, j_dict, config["feature_mode"])
    
    pair_metrics = evaluate_on_split(model, X_test, y_test)
    pair_ranking = compute_job_ndcg(test_df, pair_metrics["predicted_scores"])
    pair_metrics["ranking"] = pair_ranking
    
    print(f"Accuracy:     {pair_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro-F1:     {pair_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:  {pair_metrics['weighted_f1']:.4f}")
    print("Per-class performance:")
    for cls, vals in pair_metrics["per_class"].items():
        print(f"  {cls:15s} P={vals['precision']:.4f}  R={vals['recall']:.4f}  F1={vals['f1']:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(np.array(pair_metrics["confusion_matrix"]))
    print(f"Ranking nDCG@5: {pair_ranking['ndcg@5']:.4f} | nDCG@10: {pair_ranking['ndcg@10']:.4f} | Concordance: {pair_ranking['pairwise_concordance'] * 100:.2f}%")
    
    # Save pair test metrics
    with open(metrics_dir / "pair_held_out_test_metrics.json", "w") as f:
        # exclude raw numpy array arrays before json dumping
        clean_pair = {k: v for k, v in pair_metrics.items() if k not in ["predicted_scores", "probabilities"]}
        json.dump(clean_pair, f, indent=2)
        
    # ---------------- TEST EVALUATION 2: Zero-shot Unseen Job Descriptions ----------------
    print("\n" + "="*70)
    print("TEST EVALUATION 2: Group-Held-Out Unseen Job Descriptions (jd_held_out_test.csv)")
    print("="*70)
    jd_test_df = pd.read_csv(project_root / "artifacts" / "dataset_audit" / "splits" / "jd_held_out_test.csv")
    X_jd_test, y_jd_test = build_feature_matrix(jd_test_df, r_dict, j_dict, config["feature_mode"])
    
    jd_metrics = evaluate_on_split(model, X_jd_test, y_jd_test)
    jd_ranking = compute_job_ndcg(jd_test_df, jd_metrics["predicted_scores"])
    jd_metrics["ranking"] = jd_ranking
    
    print(f"Accuracy:     {jd_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro-F1:     {jd_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:  {jd_metrics['weighted_f1']:.4f}")
    print("Per-class performance:")
    for cls, vals in jd_metrics["per_class"].items():
        print(f"  {cls:15s} P={vals['precision']:.4f}  R={vals['recall']:.4f}  F1={vals['f1']:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(np.array(jd_metrics["confusion_matrix"]))
    print(f"Ranking nDCG@5: {jd_ranking['ndcg@5']:.4f} | nDCG@10: {jd_ranking['ndcg@10']:.4f} | Concordance: {jd_ranking['pairwise_concordance'] * 100:.2f}%")
    
    # Save jd test metrics
    with open(metrics_dir / "jd_held_out_test_metrics.json", "w") as f:
        clean_jd = {k: v for k, v in jd_metrics.items() if k not in ["predicted_scores", "probabilities"]}
        json.dump(clean_jd, f, indent=2)
        
    # Combined benchmark comparison summary
    comparison_summary = {
        "model_id": config["model_id"],
        "architecture": config["architecture"],
        "context_tokens": config["context_tokens"],
        "pair_held_out_test": {
            "rows": len(test_df),
            "accuracy": pair_metrics["accuracy"],
            "macro_f1": pair_metrics["macro_f1"],
            "weighted_f1": pair_metrics["weighted_f1"],
            "ndcg@5": pair_ranking["ndcg@5"],
            "ndcg@10": pair_ranking["ndcg@10"],
            "pairwise_concordance": pair_ranking["pairwise_concordance"]
        },
        "jd_held_out_test": {
            "rows": len(jd_test_df),
            "accuracy": jd_metrics["accuracy"],
            "macro_f1": jd_metrics["macro_f1"],
            "weighted_f1": jd_metrics["weighted_f1"],
            "ndcg@5": jd_ranking["ndcg@5"],
            "ndcg@10": jd_ranking["ndcg@10"],
            "pairwise_concordance": jd_ranking["pairwise_concordance"]
        }
    }
    with open(metrics_dir / "double_split_benchmark_summary.json", "w") as f:
        json.dump(comparison_summary, f, indent=2)
        
    print("\nBenchmark results saved successfully in:", metrics_dir)
    return comparison_summary

if __name__ == "__main__":
    run_independent_test_evaluation()
