"""
Verisift Deep Learning Accuracy Improvement Suite
Trains and validates multiple deep neural architectures and baseline models
on the 80/10/10 pair-held-out split using cached embeddings.
Model selection is performed strictly on the validation set.
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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Set reproducibility seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(14)

LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
INV_LABEL_MAP = {0: "No Fit", 1: "Potential Fit", 2: "Good Fit"}

def load_cached_embeddings(cache_path: Path):
    data = np.load(cache_path, allow_pickle=True)
    r_dict = {str(h): data["r_embeddings"][i] for i, h in enumerate(data["r_hashes"])}
    j_dict = {str(h): data["j_embeddings"][i] for i, h in enumerate(data["j_hashes"])}
    return r_dict, j_dict

def build_feature_matrix(
    df: pd.DataFrame,
    r_dict: Dict[str, np.ndarray],
    j_dict: Dict[str, np.ndarray],
    feature_mode: str = "standard"  # "standard" (1536), "rich" (1537), "augmented" (2305)
) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    
    for _, row in df.iterrows():
        r_emb = r_dict[row["r_hash"]]
        j_emb = j_dict[row["j_hash"]]
        diff = np.abs(r_emb - j_emb)
        prod = r_emb * j_emb
        cos_sim = np.dot(r_emb, j_emb)
        
        if feature_mode == "standard":
            feat = np.concatenate([r_emb, j_emb, diff, prod], axis=-1)
        elif feature_mode == "rich":
            feat = np.concatenate([r_emb, j_emb, diff, prod, [cos_sim]], axis=-1)
        elif feature_mode == "augmented":
            euclidean = np.linalg.norm(r_emb - j_emb, keepdims=True)
            sum_emb = r_emb + j_emb
            sq_diff = (r_emb - j_emb) ** 2
            feat = np.concatenate([r_emb, j_emb, diff, prod, [cos_sim], euclidean, sum_emb, sq_diff], axis=-1)
        else:
            raise ValueError(f"Unknown feature mode: {feature_mode}")
            
        features.append(feat)
        labels.append(LABEL_MAP[row["label"]])
        
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

# ----------------- Deep Learning Model Architectures -----------------

class BaselinePairwiseMLP(nn.Module):
    """Stage 4D Baseline: 1536 -> 256 -> 64 -> 3"""
    def __init__(self, input_dim: int = 1536, hidden1: int = 256, hidden2: int = 64, num_classes: int = 3, dropout1: float = 0.2, dropout2: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = self.drop1(self.act1(self.ln1(self.fc1(x))))
        out = self.drop2(self.ln2(self.fc2(out)))
        return F.gelu(out + residual)

class DeepResMLP(nn.Module):
    """
    Deep Residual MLP Architecture for High-Capacity Representation Learning:
    Input (1537) -> Project to 512 -> 2 Residual Highway Blocks -> 256 -> 64 -> 3
    """
    def __init__(self, input_dim: int = 1537, hidden_dim: int = 512, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res1 = ResidualBlock(hidden_dim, dropout=dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        h = self.input_proj(x)
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h)

class DenseCrossFeatureNet(nn.Module):
    """
    Cross-Feature Interaction Network:
    Separate branch for semantic similarity and differential projections,
    fused with gating mechanism.
    """
    def __init__(self, input_dim: int = 1537, emb_dim: int = 384, num_classes: int = 3):
        super().__init__()
        # R and J branch
        self.dual_encoder = nn.Sequential(
            nn.Linear(emb_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        # Interaction branch (|R-J|, R*J, CosSim)
        interaction_dim = emb_dim * 2 + 1 if input_dim == (emb_dim * 4 + 1) else emb_dim * 2
        self.interaction_encoder = nn.Sequential(
            nn.Linear(interaction_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        # Gating & Fusion
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )
        self.emb_dim = emb_dim
        
    def forward(self, x):
        # x is [R, J, |R-J|, R*J, cos_sim]
        r_j = x[:, :self.emb_dim * 2]
        interactions = x[:, self.emb_dim * 2:]
        h1 = self.dual_encoder(r_j)
        h2 = self.interaction_encoder(interactions)
        h = torch.cat([h1, h2], dim=-1)
        return self.fusion(h)

def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weights: torch.Tensor = None,
    patience: int = 8
) -> Tuple[nn.Module, Dict[str, Any]]:
    device = torch.device("cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    
    history = []
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
            
        train_loss /= len(X_train)
        
        # Validation
        model.eval()
        val_preds = []
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                
        val_loss /= len(X_val)
        val_preds = np.array(val_preds)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="macro")
        
        scheduler.step(val_f1)
        
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4)
        }
        history.append(record)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    elapsed = time.time() - start_time
    # Restore best weights
    model.load_state_dict(best_state)
    
    metrics = {
        "best_epoch": best_epoch,
        "total_epochs": len(history),
        "val_accuracy": round(best_val_acc, 4),
        "val_macro_f1": round(best_val_f1, 4),
        "duration_seconds": round(elapsed, 2),
        "history": history
    }
    return model, metrics

def evaluate_on_split(model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X))
            probs = F.softmax(logits, dim=1).numpy()
            preds = np.argmax(probs, axis=1)
    else:
        probs = model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    weighted_f1 = f1_score(y, preds, average="weighted")
    prec_per_class = precision_score(y, preds, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(y, preds, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y, preds, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y, preds).tolist()
    
    # Continuous ordinal relevance score for ranking: 0*P0 + 50*P1 + 100*P2
    ordinal_scores = probs[:, 1] * 50.0 + probs[:, 2] * 100.0
    
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": {
            INV_LABEL_MAP[i]: {
                "precision": round(prec_per_class[i], 4),
                "recall": round(rec_per_class[i], 4),
                "f1": round(f1_per_class[i], 4)
            } for i in range(3)
        },
        "confusion_matrix": cm,
        "predicted_scores": ordinal_scores,
        "probabilities": probs
    }

def run_experiment_suite():
    project_root = Path(__file__).resolve().parent.parent
    splits_dir = project_root / "artifacts" / "accuracy_experiments" / "splits"
    cache_dir = project_root / "artifacts" / "accuracy_experiments" / "embeddings_cache"
    checkpoints_dir = project_root / "artifacts" / "accuracy_experiments" / "checkpoints"
    metrics_dir = project_root / "artifacts" / "accuracy_experiments" / "metrics"
    
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    print("Loading data splits...")
    train_df = pd.read_csv(splits_dir / "pair_held_out_train.csv")
    val_df = pd.read_csv(splits_dir / "pair_held_out_val.csv")
    test_df = pd.read_csv(splits_dir / "pair_held_out_test.csv")
    jd_test_df = pd.read_csv(project_root / "artifacts" / "dataset_audit" / "splits" / "jd_held_out_test.csv")
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Pair Test: {len(test_df)}, JD Test: {len(jd_test_df)}")
    
    # Load cached embeddings
    print("Loading cached embeddings...")
    r_dict_256, j_dict_256 = load_cached_embeddings(cache_dir / "minilm_256_embeddings.npz")
    r_dict_512, j_dict_512 = load_cached_embeddings(cache_dir / "minilm_512_embeddings.npz")
    
    # Calculate class weights for loss balancing: inverse class frequency
    y_train_raw = np.array([LABEL_MAP[l] for l in train_df["label"]])
    class_counts = np.bincount(y_train_raw)
    class_weights = torch.tensor(len(y_train_raw) / (3.0 * class_counts), dtype=torch.float32)
    print("Class weights for balancing:", class_weights.tolist())
    
    # Prepare feature sets
    print("Building feature matrices...")
    # 256-token standard (1536d)
    X_train_256_std, y_train = build_feature_matrix(train_df, r_dict_256, j_dict_256, "standard")
    X_val_256_std, y_val = build_feature_matrix(val_df, r_dict_256, j_dict_256, "standard")
    
    # 512-token standard (1536d)
    X_train_512_std, _ = build_feature_matrix(train_df, r_dict_512, j_dict_512, "standard")
    X_val_512_std, _ = build_feature_matrix(val_df, r_dict_512, j_dict_512, "standard")
    
    # 512-token rich (1537d)
    X_train_512_rich, _ = build_feature_matrix(train_df, r_dict_512, j_dict_512, "rich")
    X_val_512_rich, _ = build_feature_matrix(val_df, r_dict_512, j_dict_512, "rich")
    
    # 512-token augmented (2305d)
    X_train_512_aug, _ = build_feature_matrix(train_df, r_dict_512, j_dict_512, "augmented")
    X_val_512_aug, _ = build_feature_matrix(val_df, r_dict_512, j_dict_512, "augmented")
    
    print(f"X_train_256_std shape: {X_train_256_std.shape}")
    print(f"X_train_512_rich shape: {X_train_512_rich.shape}")
    print(f"X_train_512_aug shape: {X_train_512_aug.shape}")
    
    experiment_results = []
    trained_models = {}
    
    # ---------------- EXPERIMENT 1: Baseline Pairwise MLP (256 tokens) ----------------
    print("\n" + "="*70)
    print("EXPERIMENT 1: Baseline Pairwise MLP (256-token MiniLM, 1536d)")
    print("="*70)
    m1 = BaselinePairwiseMLP(input_dim=1536)
    m1, res1 = train_torch_model(m1, X_train_256_std, y_train, X_val_256_std, y_val, epochs=35, batch_size=64, lr=1e-3)
    val_metrics1 = evaluate_on_split(m1, X_val_256_std, y_val)
    print(f"-> Val Acc: {val_metrics1['accuracy'] * 100:.2f}%, Val Macro-F1: {val_metrics1['macro_f1']:.4f}")
    trained_models["M1_Baseline_MLP_256"] = (m1, "standard", 256)
    experiment_results.append({
        "model_id": "M1_Baseline_MLP_256",
        "description": "Baseline Pairwise MLP (1536d) with 256-token MiniLM embeddings",
        "val_accuracy": val_metrics1["accuracy"],
        "val_macro_f1": val_metrics1["macro_f1"],
        "val_weighted_f1": val_metrics1["weighted_f1"],
        "duration_sec": res1["duration_seconds"],
        "best_epoch": res1["best_epoch"]
    })
    
    # ---------------- EXPERIMENT 2: Baseline Pairwise MLP (512 tokens) ----------------
    print("\n" + "="*70)
    print("EXPERIMENT 2: Baseline Pairwise MLP with Extended Context (512-token MiniLM, 1536d)")
    print("="*70)
    m2 = BaselinePairwiseMLP(input_dim=1536)
    m2, res2 = train_torch_model(m2, X_train_512_std, y_train, X_val_512_std, y_val, epochs=35, batch_size=64, lr=1e-3)
    val_metrics2 = evaluate_on_split(m2, X_val_512_std, y_val)
    print(f"-> Val Acc: {val_metrics2['accuracy'] * 100:.2f}%, Val Macro-F1: {val_metrics2['macro_f1']:.4f}")
    trained_models["M2_Baseline_MLP_512"] = (m2, "standard", 512)
    experiment_results.append({
        "model_id": "M2_Baseline_MLP_512",
        "description": "Pairwise MLP (1536d) with 512-token MiniLM embeddings (extended context)",
        "val_accuracy": val_metrics2["accuracy"],
        "val_macro_f1": val_metrics2["macro_f1"],
        "val_weighted_f1": val_metrics2["weighted_f1"],
        "duration_sec": res2["duration_seconds"],
        "best_epoch": res2["best_epoch"]
    })
    
    # ---------------- EXPERIMENT 3: Rich Interaction MLP (512 tokens, 1537d) ----------------
    print("\n" + "="*70)
    print("EXPERIMENT 3: Rich Interaction MLP (512-token, 1537d with CosSim)")
    print("="*70)
    m3 = BaselinePairwiseMLP(input_dim=1537)
    m3, res3 = train_torch_model(m3, X_train_512_rich, y_train, X_val_512_rich, y_val, epochs=35, batch_size=64, lr=8e-4)
    val_metrics3 = evaluate_on_split(m3, X_val_512_rich, y_val)
    print(f"-> Val Acc: {val_metrics3['accuracy'] * 100:.2f}%, Val Macro-F1: {val_metrics3['macro_f1']:.4f}")
    trained_models["M3_Rich_MLP_512"] = (m3, "rich", 512)
    experiment_results.append({
        "model_id": "M3_Rich_MLP_512",
        "description": "Rich Interaction MLP (1537d) with 512-token embeddings + CosSim",
        "val_accuracy": val_metrics3["accuracy"],
        "val_macro_f1": val_metrics3["macro_f1"],
        "val_weighted_f1": val_metrics3["weighted_f1"],
        "duration_sec": res3["duration_seconds"],
        "best_epoch": res3["best_epoch"]
    })
    
    # ---------------- EXPERIMENT 4: Deep Residual MLP (DeepResMLP, 1537d) ----------------
    print("\n" + "="*70)
    print("EXPERIMENT 4: Deep Residual MLP (DeepResMLP, 512h, 2 ResBlocks, GELU, 1537d)")
    print("="*70)
    m4 = DeepResMLP(input_dim=1537, hidden_dim=512, dropout=0.2)
    m4, res4 = train_torch_model(m4, X_train_512_rich, y_train, X_val_512_rich, y_val, epochs=45, batch_size=64, lr=6e-4)
    val_metrics4 = evaluate_on_split(m4, X_val_512_rich, y_val)
    print(f"-> Val Acc: {val_metrics4['accuracy'] * 100:.2f}%, Val Macro-F1: {val_metrics4['macro_f1']:.4f}")
    trained_models["M4_DeepResMLP_512"] = (m4, "rich", 512)
    experiment_results.append({
        "model_id": "M4_DeepResMLP_512",
        "description": "Deep Residual MLP (512h, 2 ResBlocks, GELU, LayerNorm, Dropout 0.2)",
        "val_accuracy": val_metrics4["accuracy"],
        "val_macro_f1": val_metrics4["macro_f1"],
        "val_weighted_f1": val_metrics4["weighted_f1"],
        "duration_sec": res4["duration_seconds"],
        "best_epoch": res4["best_epoch"]
    })
    
    # ---------------- EXPERIMENT 5: Dense Cross-Feature Network (1537d) ----------------
    print("\n" + "="*70)
    print("EXPERIMENT 5: Dense Cross-Feature Network (Dual-branch cross-interaction, 1537d)")
    print("="*70)
    m5 = DenseCrossFeatureNet(input_dim=1537, emb_dim=384)
    m5, res5 = train_torch_model(m5, X_train_512_rich, y_train, X_val_512_rich, y_val, epochs=40, batch_size=64, lr=8e-4)
    val_metrics5 = evaluate_on_split(m5, X_val_512_rich, y_val)
    print(f"-> Val Acc: {val_metrics5['accuracy'] * 100:.2f}%, Val Macro-F1: {val_metrics5['macro_f1']:.4f}")
    trained_models["M5_DenseCrossNet_512"] = (m5, "rich", 512)
    experiment_results.append({
        "model_id": "M5_DenseCrossNet_512",
        "description": "Dense Cross-Feature Network (Dual-branch representation + interaction fusion)",
        "val_accuracy": val_metrics5["accuracy"],
        "val_macro_f1": val_metrics5["macro_f1"],
        "val_weighted_f1": val_metrics5["weighted_f1"],
        "duration_sec": res5["duration_seconds"],
        "best_epoch": res5["best_epoch"]
    })
    
    # ---------------- EXPERIMENT 6: Histogram Gradient Boosting (Tabular Baseline) ----------------
    print("\n" + "="*70)
    print("EXPERIMENT 6: Histogram Gradient Boosting (1537d Tabular Benchmark)")
    print("="*70)
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=6, random_state=SEED)
    hgb.fit(X_train_512_rich, y_train)
    val_metrics6 = evaluate_on_split(hgb, X_val_512_rich, y_val)
    hgb_duration = round(time.time() - t0, 2)
    print(f"-> Val Acc: {val_metrics6['accuracy'] * 100:.2f}%, Val Macro-F1: {val_metrics6['macro_f1']:.4f}")
    trained_models["M6_HistGradientBoosting"] = (hgb, "rich", 512)
    experiment_results.append({
        "model_id": "M6_HistGradientBoosting",
        "description": "Histogram Gradient Boosting Classifier (150 trees, max_depth 6)",
        "val_accuracy": val_metrics6["accuracy"],
        "val_macro_f1": val_metrics6["macro_f1"],
        "val_weighted_f1": val_metrics6["weighted_f1"],
        "duration_sec": hgb_duration,
        "best_epoch": 150
    })

    # ---------------- EXPERIMENT 7: Regularized DeepResMLP with Class Balancing (1537d) ----------------
    print("\n" + "="*70)
    print("EXPERIMENT 7: Weighted Regularized DeepResMLP (Class-Weighted Loss + Tuned Dropout)")
    print("="*70)
    m7 = DeepResMLP(input_dim=1537, hidden_dim=512, dropout=0.25)
    # Give moderate boost to minority classes
    mild_weights = torch.tensor([0.85, 1.15, 1.15], dtype=torch.float32)
    m7, res7 = train_torch_model(m7, X_train_512_rich, y_train, X_val_512_rich, y_val, epochs=45, batch_size=64, lr=5e-4, weight_decay=2e-4, class_weights=mild_weights)
    val_metrics7 = evaluate_on_split(m7, X_val_512_rich, y_val)
    print(f"-> Val Acc: {val_metrics7['accuracy'] * 100:.2f}%, Val Macro-F1: {val_metrics7['macro_f1']:.4f}")
    trained_models["M7_Weighted_DeepResMLP_512"] = (m7, "rich", 512)
    experiment_results.append({
        "model_id": "M7_Weighted_DeepResMLP_512",
        "description": "DeepResMLP with Class-Weighted Loss (mild 0.85/1.15/1.15) & Weight Decay 2e-4",
        "val_accuracy": val_metrics7["accuracy"],
        "val_macro_f1": val_metrics7["macro_f1"],
        "val_weighted_f1": val_metrics7["weighted_f1"],
        "duration_sec": res7["duration_seconds"],
        "best_epoch": res7["best_epoch"]
    })
    
    # ---------------- SUMMARY & MODEL SELECTION ----------------
    print("\n" + "#"*70)
    print("VALIDATION MODEL SELECTION SUMMARY")
    print("#"*70)
    df_results = pd.DataFrame(experiment_results).sort_values(by=["val_macro_f1", "val_accuracy"], ascending=False)
    print(df_results.to_string(index=False))
    
    # Save validation results
    df_results.to_json(metrics_dir / "validation_model_selection.json", orient="records", indent=2)
    
    best_model_id = df_results.iloc[0]["model_id"]
    print(f"\nWINNER SELECTED BASED ON VALIDATION SET: {best_model_id}")
    
    best_model_obj, best_feature_mode, best_context = trained_models[best_model_id]
    
    # Save best model checkpoint
    if isinstance(best_model_obj, nn.Module):
        torch.save(best_model_obj.state_dict(), checkpoints_dir / "best_deep_model.pt")
        config = {
            "model_id": best_model_id,
            "architecture": type(best_model_obj).__name__,
            "feature_mode": best_feature_mode,
            "context_tokens": best_context,
            "input_dim": 1537 if best_feature_mode == "rich" else 1536,
            "val_accuracy": float(df_results.iloc[0]["val_accuracy"]),
            "val_macro_f1": float(df_results.iloc[0]["val_macro_f1"])
        }
        with open(checkpoints_dir / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved best model checkpoint to {checkpoints_dir / 'best_deep_model.pt'}")
    
    return trained_models, df_results

if __name__ == "__main__":
    run_experiment_suite()

