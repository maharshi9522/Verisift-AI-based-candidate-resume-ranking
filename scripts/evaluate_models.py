"""
Verisift Comprehensive Benchmark & Multi-Model Evaluator
Evaluates on identical held-out test splits:
1. Stage 4C TF-IDF Baseline
2. Existing Pretrained MiniLM + Pairwise MLP (E4)
3. Fine-Tuned SBERT (Joint Encoder)
4. Final Explainability & Section-Aware Ranking System
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import ndcg_score

# Add project_root and backend to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from app.services.text_preprocessor import TextPreprocessor
from app.services.resume_parser import ResumeParser
from app.services.explainability_service import ExplainabilityService

LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
ORDINAL_WEIGHTS = np.array([0.0, 0.5, 1.0])

def compute_pairwise_features_np(r_emb: np.ndarray, j_emb: np.ndarray) -> np.ndarray:
    diff = np.abs(r_emb - j_emb)
    prod = r_emb * j_emb
    return np.concatenate([r_emb, j_emb, diff, prod], axis=-1)

def compute_pairwise_concordance(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, int, int, int]:
    """
    Computes pairwise ranking concordance across pairs where y_true[i] != y_true[j].
    Returns: (concordance_ratio, concordant, discordant, ties)
    """
    n = len(y_true)
    concordant = 0
    discordant = 0
    ties = 0
    total = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            total += 1
            if (y_true[i] > y_true[j] and y_score[i] > y_score[j]) or (y_true[i] < y_true[j] and y_score[i] < y_score[j]):
                concordant += 1
            elif y_score[i] == y_score[j]:
                ties += 1
            else:
                discordant += 1
                
    ratio = concordant / total if total > 0 else 0.0
    return ratio, concordant, discordant, ties

def evaluate_job_ranking_metrics(
    df: pd.DataFrame,
    pred_scores: np.ndarray,
    jd_col: str = "j_hash",
    label_col: str = "label"
) -> Dict[str, Any]:
    """
    Evaluates nDCG@5, nDCG@10, Precision@5, Recall@5, and Pairwise Concordance per job requisition.
    """
    df_eval = df.copy()
    df_eval["score"] = pred_scores
    df_eval["true_num"] = df_eval[label_col].map(LABEL_MAP)
    
    ndcg5_list = []
    ndcg10_list = []
    p5_list = []
    r5_list = []
    total_concordant = 0
    total_discordant = 0
    total_tied = 0
    evaluated_jobs = 0
    
    for j_val, group in df_eval.groupby(jd_col):
        if len(group) < 2:
            continue
        # IDCG check: need at least one positive candidate
        if group["true_num"].max() == 0:
            continue
            
        y_true = group["true_num"].to_numpy().reshape(1, -1)
        y_pred = group["score"].to_numpy().reshape(1, -1)
        
        try:
            n5 = ndcg_score(y_true, y_pred, k=5)
            n10 = ndcg_score(y_true, y_pred, k=10)
            ndcg5_list.append(n5)
            ndcg10_list.append(n10)
        except Exception:
            continue
            
        # Sort candidates descending
        sorted_group = group.sort_values(by="score", ascending=False)
        top5 = sorted_group.head(5)
        pos_in_top5 = (top5["true_num"] > 0).sum()
        total_pos_in_job = (group["true_num"] > 0).sum()
        
        p5_list.append(pos_in_top5 / min(5, len(group)))
        r5_list.append(pos_in_top5 / total_pos_in_job if total_pos_in_job > 0 else 0.0)
        
        _, c, d, t = compute_pairwise_concordance(group["true_num"].to_numpy(), group["score"].to_numpy())
        total_concordant += c
        total_discordant += d
        total_tied += t
        evaluated_jobs += 1
        
    total_pairs = total_concordant + total_discordant + total_tied
    concordance = total_concordant / total_pairs if total_pairs > 0 else 0.0
    
    return {
        "evaluated_jobs": evaluated_jobs,
        "ndcg5_mean": float(np.mean(ndcg5_list)) if ndcg5_list else 0.0,
        "ndcg5_std": float(np.std(ndcg5_list)) if ndcg5_list else 0.0,
        "ndcg10_mean": float(np.mean(ndcg10_list)) if ndcg10_list else 0.0,
        "ndcg10_std": float(np.std(ndcg10_list)) if ndcg10_list else 0.0,
        "precision5_mean": float(np.mean(p5_list)) if p5_list else 0.0,
        "recall5_mean": float(np.mean(r5_list)) if r5_list else 0.0,
        "pairwise_concordance": float(concordance),
        "concordant_pairs": total_concordant,
        "discordant_pairs": total_discordant,
        "tied_pairs": total_tied,
        "total_pairs": total_pairs
    }

def mean_pool_emb(model, tokenizer, texts, device, max_length=256):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        pooled = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        normed = F.normalize(pooled, p=2, dim=1)
    return normed.cpu().numpy()

def run_evaluation(test_csv_path: Path, output_dir: Path, split_name: str = "resume_held_out"):
    output_dir.mkdir(parents=True, exist_ok=True)
    df_test = pd.read_csv(test_csv_path)
    print(f"\n=======================================================")
    print(f"RUNNING BENCHMARK EVALUATION ON {split_name.upper()} ({len(df_test)} rows)")
    print(f"=======================================================")
    
    y_true_labels = df_test["label"].tolist()
    y_true_indices = np.array([LABEL_MAP[lbl] for lbl in y_true_labels])
    
    results = {}
    
    # -------------------------------------------------------------
    # MODEL 1: Existing Pretrained MiniLM + MLP (Frozen E4 Baseline)
    # -------------------------------------------------------------
    print("\n--- Evaluating Model 1: Existing Pretrained MiniLM + E4 MLP ---")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    base_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    
    from app.services.matching_service import PairwiseMLP
    old_mlp = PairwiseMLP(input_dim=1536, hidden_dim1=256, hidden_dim2=64, num_classes=3)
    old_weights_path = project_root / "artifacts" / "semantic_ranker" / "semantic_ranker_mlp.pt"
    old_mlp.load_state_dict(torch.load(old_weights_path, map_location="cpu"))
    old_mlp.eval()
    
    # Encode unique JDs and Resumes
    unique_jds = df_test["job_description_text"].unique()
    unique_resumes = df_test["resume_text"].unique()
    
    jd_clean_map = {jd: TextPreprocessor.normalize(jd) for jd in unique_jds}
    res_clean_map = {res: TextPreprocessor.normalize(res) for res in unique_resumes}
    
    jd_embs = base_encoder.encode([jd_clean_map[jd] for jd in unique_jds], normalize_embeddings=True)
    jd_emb_dict = {jd: emb for jd, emb in zip(unique_jds, jd_embs)}
    
    res_embs = base_encoder.encode([res_clean_map[res] for res in unique_resumes], normalize_embeddings=True)
    res_emb_dict = {res: emb for res, emb in zip(unique_resumes, res_embs)}
    
    # Predict all rows
    m1_probs = []
    m1_scores = []
    for _, row in df_test.iterrows():
        r_emb = res_emb_dict[row["resume_text"]]
        j_emb = jd_emb_dict[row["job_description_text"]]
        pair_feat = compute_pairwise_features_np(r_emb, j_emb)
        with torch.no_grad():
            logits = old_mlp(torch.tensor(pair_feat, dtype=torch.float32).unsqueeze(0))
            prob = torch.softmax(logits, dim=1).squeeze(0).numpy()
            prob = np.nan_to_num(prob, nan=0.0)
            if prob.sum() > 0:
                prob = prob / prob.sum()
            else:
                prob = np.array([1.0, 0.0, 0.0])
        score = 100.0 * np.dot(ORDINAL_WEIGHTS, prob)
        m1_probs.append(prob)
        m1_scores.append(score)
        
    m1_probs = np.array(m1_probs)
    m1_scores = np.array(m1_scores)
    m1_preds = np.argmax(m1_probs, axis=1)
    m1_time = time.time() - t0
    
    m1_acc = accuracy_score(y_true_indices, m1_preds)
    m1_f1 = f1_score(y_true_indices, m1_preds, average="macro")
    m1_wf1 = f1_score(y_true_indices, m1_preds, average="weighted")
    m1_prec = precision_score(y_true_indices, m1_preds, average=None)
    m1_rec = recall_score(y_true_indices, m1_preds, average=None)
    m1_f1s = f1_score(y_true_indices, m1_preds, average=None)
    m1_cm = confusion_matrix(y_true_indices, m1_preds).tolist()
    m1_rank = evaluate_job_ranking_metrics(df_test, m1_scores)
    
    results["Pretrained MiniLM + MLP (Baseline)"] = {
        "accuracy": float(m1_acc),
        "macro_f1": float(m1_f1),
        "weighted_f1": float(m1_wf1),
        "per_class_precision": [float(p) for p in m1_prec],
        "per_class_recall": [float(r) for r in m1_rec],
        "per_class_f1": [float(f) for f in m1_f1s],
        "confusion_matrix": m1_cm,
        "ranking": m1_rank,
        "latency_sec": float(m1_time)
    }
    print(f"M1 Result: Acc={m1_acc*100:.2f}% | Macro-F1={m1_f1:.4f} | nDCG@5={m1_rank['ndcg5_mean']:.4f} | Concordance={m1_rank['pairwise_concordance']*100:.2f}%")
    
    # -------------------------------------------------------------
    # MODEL 2: Fine-Tuned SBERT Model
    # -------------------------------------------------------------
    print("\n--- Evaluating Model 2: Fine-Tuned SBERT Dual-Encoder ---")
    t0 = time.time()
    ft_dir = project_root / "artifacts" / "finetuned_sbert"
    ft_encoder_path = ft_dir / "sbert_encoder"
    ft_head_path = ft_dir / "matching_head.pt"
    
    if ft_encoder_path.exists() and ft_head_path.exists():
        ft_tokenizer = AutoTokenizer.from_pretrained(ft_encoder_path)
        ft_encoder = AutoModel.from_pretrained(ft_encoder_path)
        ft_encoder.eval()
        
        # Load head
        from scripts.train_finetuned_sbert import JointSBERTMatchingModel
        temp_model = JointSBERTMatchingModel()
        temp_model.classifier.load_state_dict(torch.load(ft_head_path, map_location="cpu"))
        ft_classifier = temp_model.classifier
        ft_classifier.eval()
        
        # Encode unique JDs and Resumes with fine-tuned encoder
        ft_jd_embs = mean_pool_emb(ft_encoder, ft_tokenizer, [jd_clean_map[jd] for jd in unique_jds], "cpu", 256)
        ft_jd_emb_dict = {jd: emb for jd, emb in zip(unique_jds, ft_jd_embs)}
        
        ft_res_embs = mean_pool_emb(ft_encoder, ft_tokenizer, [res_clean_map[res] for res in unique_resumes], "cpu", 256)
        ft_res_emb_dict = {res: emb for res, emb in zip(unique_resumes, ft_res_embs)}
        
        m2_probs = []
        m2_scores = []
        for _, row in df_test.iterrows():
            r_emb = ft_res_emb_dict[row["resume_text"]]
            j_emb = ft_jd_emb_dict[row["job_description_text"]]
            pair_feat = compute_pairwise_features_np(r_emb, j_emb)
            with torch.no_grad():
                logits = ft_classifier(torch.tensor(pair_feat, dtype=torch.float32).unsqueeze(0))
                prob = torch.softmax(logits, dim=1).squeeze(0).numpy()
                prob = np.nan_to_num(prob, nan=0.0)
                if prob.sum() > 0:
                    prob = prob / prob.sum()
                else:
                    prob = np.array([1.0, 0.0, 0.0])
            score = 100.0 * np.dot(ORDINAL_WEIGHTS, prob)
            m2_probs.append(prob)
            m2_scores.append(score)
            
        m2_probs = np.array(m2_probs)
        m2_scores = np.array(m2_scores)
        m2_preds = np.argmax(m2_probs, axis=1)
        m2_time = time.time() - t0
        
        m2_acc = accuracy_score(y_true_indices, m2_preds)
        m2_f1 = f1_score(y_true_indices, m2_preds, average="macro")
        m2_wf1 = f1_score(y_true_indices, m2_preds, average="weighted")
        m2_prec = precision_score(y_true_indices, m2_preds, average=None)
        m2_rec = recall_score(y_true_indices, m2_preds, average=None)
        m2_f1s = f1_score(y_true_indices, m2_preds, average=None)
        m2_cm = confusion_matrix(y_true_indices, m2_preds).tolist()
        m2_rank = evaluate_job_ranking_metrics(df_test, m2_scores)
        
        results["Fine-Tuned SBERT Dual-Encoder"] = {
            "accuracy": float(m2_acc),
            "macro_f1": float(m2_f1),
            "weighted_f1": float(m2_wf1),
            "per_class_precision": [float(p) for p in m2_prec],
            "per_class_recall": [float(r) for r in m2_rec],
            "per_class_f1": [float(f) for f in m2_f1s],
            "confusion_matrix": m2_cm,
            "ranking": m2_rank,
            "latency_sec": float(m2_time)
        }
        print(f"M2 Result: Acc={m2_acc*100:.2f}% | Macro-F1={m2_f1:.4f} | nDCG@5={m2_rank['ndcg5_mean']:.4f} | Concordance={m2_rank['pairwise_concordance']*100:.2f}%")
    else:
        print("Fine-tuned SBERT artifacts not yet found; skipping M2.")

    # -------------------------------------------------------------
    # MODEL 3: Final Explainability & Section-Aware Ranking System
    # -------------------------------------------------------------
    print("\n--- Evaluating Model 3: Final Explainability & Section-Aware System ---")
    t0 = time.time()
    # Combines fine-tuned semantic score with evidence features
    m3_scores = []
    # If M2 available use M2 scores as semantic component, else M1
    base_sem = m2_scores if len(results) > 1 else m1_scores
    
    for idx, row in df_test.iterrows():
        sem_score = float(base_sem[idx])
        # Generate lightweight explainability assessment
        # Parse resume into structured format
        parsed = ResumeParser.parse_resume(row["resume_text"])
        struct_dict = parsed.to_dict()
        
        expl = ExplainabilityService.generate_candidate_explanation(
            candidate_name="Candidate",
            job_title="Job Role",
            job_description=row["job_description_text"],
            resume_text=row["resume_text"],
            semantic_score=sem_score,
            structured_resume=struct_dict
        )
        m3_scores.append(expl.relevance_score)
        
    m3_scores = np.array(m3_scores)
    # Predict category based on calibrated thresholds: >=70 Good Fit, >=45 Potential Fit, else No Fit
    m3_preds = np.where(m3_scores >= 70.0, 2, np.where(m3_scores >= 45.0, 1, 0))
    m3_time = time.time() - t0
    
    m3_acc = accuracy_score(y_true_indices, m3_preds)
    m3_f1 = f1_score(y_true_indices, m3_preds, average="macro")
    m3_wf1 = f1_score(y_true_indices, m3_preds, average="weighted")
    m3_prec = precision_score(y_true_indices, m3_preds, average=None)
    m3_rec = recall_score(y_true_indices, m3_preds, average=None)
    m3_f1s = f1_score(y_true_indices, m3_preds, average=None)
    m3_cm = confusion_matrix(y_true_indices, m3_preds).tolist()
    m3_rank = evaluate_job_ranking_metrics(df_test, m3_scores)
    
    results["Final Explainable Ranking System"] = {
        "accuracy": float(m3_acc),
        "macro_f1": float(m3_f1),
        "weighted_f1": float(m3_wf1),
        "per_class_precision": [float(p) for p in m3_prec],
        "per_class_recall": [float(r) for r in m3_rec],
        "per_class_f1": [float(f) for f in m3_f1s],
        "confusion_matrix": m3_cm,
        "ranking": m3_rank,
        "latency_sec": float(m3_time)
    }
    print(f"M3 Result: Acc={m3_acc*100:.2f}% | Macro-F1={m3_f1:.4f} | nDCG@5={m3_rank['ndcg5_mean']:.4f} | Concordance={m3_rank['pairwise_concordance']*100:.2f}%")
    
    # Save benchmark results
    out_file = output_dir / f"benchmark_comparison_{split_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark metrics to: {out_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="resume_held_out")
    args = parser.parse_args()
    
    splits_dir = project_root / "artifacts" / "dataset_audit" / "splits"
    if args.split == "resume_held_out":
        test_file = splits_dir / "resume_held_out_test.csv"
    elif args.split == "jd_held_out":
        test_file = splits_dir / "jd_held_out_test.csv"
    else:
        test_file = splits_dir / "combined_held_out_test.csv"
        
    out_dir = project_root / "artifacts" / "model_comparison"
    run_evaluation(test_file, out_dir, split_name=args.split)
