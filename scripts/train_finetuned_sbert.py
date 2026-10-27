"""
Verisift SBERT Parameter-Efficient Fine-Tuning Pipeline
Fine-tunes the upper semantic matching layers (Layers 4 & 5) of the SBERT encoder
(sentence-transformers/all-MiniLM-L6-v2) jointly with the 1,536-dimensional
interaction classification head on resume-job relevance pairs.
Verifies that transformer encoder weights genuinely update during training.
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

LABEL_MAPPING = {
    "No Fit": 0,
    "Potential Fit": 1,
    "Good Fit": 2
}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

class ResumeJobFitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 192):
        self.resumes = df["resume_text"].tolist()
        self.jobs = df["job_description_text"].tolist()
        self.labels = [LABEL_MAPPING[lbl] for lbl in df["label"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "resume_text": self.resumes[idx],
            "job_text": self.jobs[idx],
            "label": self.labels[idx]
        }

def collate_fn(batch, tokenizer, max_length=192):
    resumes = [item["resume_text"] for item in batch]
    jobs = [item["job_text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    r_enc = tokenizer(
        resumes,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    j_enc = tokenizer(
        jobs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        "r_input_ids": r_enc["input_ids"],
        "r_attention_mask": r_enc["attention_mask"],
        "j_input_ids": j_enc["input_ids"],
        "j_attention_mask": j_enc["attention_mask"],
        "labels": labels
    }

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class JointSBERTMatchingModel(nn.Module):
    """
    Joint Dual-Encoder Architecture with Parameter-Efficient Fine-Tuning:
    - Base transformer: all-MiniLM-L6-v2 (384-dim).
    - Lower layers (0-3) frozen; Upper semantic matching layers (4-5) fine-tuned.
    - Mean pooling + L2 normalization -> R, J in R^384.
    - Pairwise feature interaction: z = [R, J, |R - J|, R * J] in R^1536.
    - Multi-class classification head: Linear(1536, 256) -> LayerNorm -> ReLU -> Dropout(0.2) -> Linear(256, 64) -> LayerNorm -> ReLU -> Dropout(0.1) -> Linear(64, 3).
    """
    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2", hidden_dim1: int = 256, hidden_dim2: int = 64, tune_top_k_layers: int = 2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Freeze lower layers, unfreeze top k layers
        tunable_layer_names = [f"layer.{5 - i}" for i in range(tune_top_k_layers)]
        for name, param in self.encoder.named_parameters():
            if any(tl in name for tl in tunable_layer_names) or "pooler" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        emb_dim = self.encoder.config.hidden_size # 384
        input_dim = emb_dim * 4 # 1536
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim2, 3)
        )

    def encode_text(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=1)

    def forward(self, r_input_ids, r_attention_mask, j_input_ids, j_attention_mask):
        r_emb = self.encode_text(r_input_ids, r_attention_mask)
        j_emb = self.encode_text(j_input_ids, j_attention_mask)

        diff = torch.abs(r_emb - j_emb)
        prod = r_emb * j_emb
        z = torch.cat([r_emb, j_emb, diff, prod], dim=-1)

        logits = self.classifier(z)
        return logits, r_emb, j_emb

def train_sbert(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 16,
    encoder_lr: float = 3e-5,
    classifier_lr: float = 8e-4,
    max_length: int = 192,
    max_train_samples: int = 800,
    max_val_samples: int = 250,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    torch.set_num_threads(14)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits_dir = data_dir / "splits"
    train_csv = splits_dir / "resume_held_out_train.csv"
    val_csv = splits_dir / "resume_held_out_val.csv"
    
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Missing splits in {splits_dir}. Run dataset_pipeline.py first.")
        
    df_train_full = pd.read_csv(train_csv)
    df_val_full = pd.read_csv(val_csv)
    
    # Stratified representative sampling to keep training time resource-conscious
    train_subsets = []
    for lbl, group in df_train_full.groupby("label"):
        n = max(1, int(max_train_samples * len(group) / len(df_train_full)))
        train_subsets.append(group.sample(n=min(len(group), n), random_state=seed))
    df_train = pd.concat(train_subsets, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    val_subsets = []
    for lbl, group in df_val_full.groupby("label"):
        n = max(1, int(max_val_samples * len(group) / len(df_val_full)))
        val_subsets.append(group.sample(n=min(len(group), n), random_state=seed))
    df_val = pd.concat(val_subsets, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    print(f"Sampled train set: {len(df_train)} rows | val set: {len(df_val)} rows.", flush=True)
    print("Train class distribution:", df_train["label"].value_counts().to_dict(), flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    train_dataset = ResumeJobFitDataset(df_train, tokenizer, max_length=max_length)
    val_dataset = ResumeJobFitDataset(df_val, tokenizer, max_length=max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=max_length)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=max_length)
    )
    
    # Initialize joint model
    model = JointSBERTMatchingModel("sentence-transformers/all-MiniLM-L6-v2", tune_top_k_layers=2)
    model.to(device)
    
    # Snapshot initial encoder weights to verify genuine weight updates
    initial_weights_snapshot = {}
    for name, param in model.encoder.named_parameters():
        if param.requires_grad:
            initial_weights_snapshot[name] = param.detach().cpu().clone()
            
    print(f"Number of trainable encoder parameters: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad):,}", flush=True)
    print(f"Number of trainable classifier parameters: {sum(p.numel() for p in model.classifier.parameters() if p.requires_grad):,}", flush=True)
    
    # Class weights for loss
    class_counts = df_train["label"].value_counts()
    n_samples = len(df_train)
    weights = [
        n_samples / (3.0 * class_counts.get("No Fit", 1)),
        n_samples / (3.0 * class_counts.get("Potential Fit", 1)),
        n_samples / (3.0 * class_counts.get("Good Fit", 1))
    ]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Differential learning rates
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": encoder_lr, "weight_decay": 0.01},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": classifier_lr, "weight_decay": 0.001}
    ])
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "epoch_times_sec": []
    }
    
    best_val_macro_f1 = -1.0
    best_epoch = -1
    start_total_time = time.time()
    
    print("\n" + "=" * 70, flush=True)
    print(f"STARTING SBERT PARAMETER-EFFICIENT JOINT FINE-TUNING ({epochs} Epochs, {len(train_loader)} steps/epoch)", flush=True)
    print("=" * 70, flush=True)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        step_count = 0
        
        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            
            r_ids = batch["r_input_ids"].to(device)
            r_mask = batch["r_attention_mask"].to(device)
            j_ids = batch["j_input_ids"].to(device)
            j_mask = batch["j_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits, _, _ = model(r_ids, r_mask, j_ids, j_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            step_count += 1
            
            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                print(f"Epoch {epoch}/{epochs} | Step {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Avg: {train_loss / step_count:.4f}", flush=True)
                
        avg_train_loss = train_loss / step_count
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                r_ids = batch["r_input_ids"].to(device)
                r_mask = batch["r_attention_mask"].to(device)
                j_ids = batch["j_input_ids"].to(device)
                j_mask = batch["j_attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                logits, _, _ = model(r_ids, r_mask, j_ids, j_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_steps += 1
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / val_steps
        val_acc = accuracy_score(all_labels, all_preds)
        val_macro_f1 = f1_score(all_labels, all_preds, average="macro")
        epoch_dur = time.time() - t0
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["epoch_times_sec"].append(epoch_dur)
        
        print(f"\n---> Epoch {epoch} Results in {epoch_dur:.1f}s:", flush=True)
        print(f"     Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_macro_f1:.4f}", flush=True)
        
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            
            print(f"     *** New best validation Macro-F1: {best_val_macro_f1:.4f}! Saving checkpoint...", flush=True)
            encoder_save_dir = output_dir / "sbert_encoder"
            model.encoder.save_pretrained(encoder_save_dir)
            tokenizer.save_pretrained(encoder_save_dir)
            
            torch.save(model.classifier.state_dict(), output_dir / "matching_head.pt")
            torch.save(model.state_dict(), output_dir / "full_joint_model.pt")
            
    total_training_time = time.time() - start_total_time
    print("\n" + "=" * 70, flush=True)
    print(f"TRAINING COMPLETE in {total_training_time:.1f}s. Best Epoch: {best_epoch} with Val Macro-F1: {best_val_macro_f1:.4f}", flush=True)
    print("=" * 70, flush=True)
    
    # -------------------------------------------------------------------------
    # VERIFY ENCODER WEIGHT UPDATES (Prove weights actually updated)
    # -------------------------------------------------------------------------
    best_encoder = AutoModel.from_pretrained(output_dir / "sbert_encoder")
    weight_deltas = {}
    for name, initial_w in initial_weights_snapshot.items():
        trained_w = best_encoder.state_dict()[name]
        l2_diff = float(torch.norm(trained_w - initial_w).item())
        weight_deltas[name] = l2_diff
        
    avg_l2_delta = float(np.mean(list(weight_deltas.values())))
    print(f"\nEncoder Weight Update Verification:", flush=True)
    print(f"Average L2 Parameter Delta in Fine-Tuned Layers: {avg_l2_delta:.6f}", flush=True)
    assert avg_l2_delta > 1e-4, f"FAIL: Encoder weights did NOT update during training! (delta={avg_l2_delta})"
    print("SUCCESS: Transformer encoder weights genuinely updated during fine-tuning!", flush=True)
    
    # Save metadata manifest
    manifest = {
        "model_name": "Verisift-FineTuned-SBERT-v2.0",
        "base_encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "architecture": {
            "encoder": "6-layer MiniLM transformer (layers 4-5 fine-tuned, layers 0-3 frozen)",
            "pooling": "Mean Pooling + L2 Normalization (384-dim)",
            "features": "Concatenated [R, J, |R - J|, R * J] (1536-dim)",
            "classification_head": "Linear(1536, 256) -> LayerNorm -> ReLU -> Dropout(0.2) -> Linear(256, 64) -> LayerNorm -> ReLU -> Dropout(0.1) -> Linear(64, 3)"
        },
        "training_environment": {
            "device": "cpu",
            "threads": 14,
            "hardware": "Intel Core Ultra 5 125H, 16 GB RAM",
            "pytorch_version": torch.__version__,
            "transformers_version": "5.20.2"
        },
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "encoder_lr": encoder_lr,
            "classifier_lr": classifier_lr,
            "max_length": max_length,
            "max_train_samples": max_train_samples,
            "random_seed": seed,
            "class_weights": weights
        },
        "results": {
            "best_epoch": best_epoch,
            "best_val_macro_f1": float(best_val_macro_f1),
            "total_training_time_sec": round(total_training_time, 2),
            "encoder_weight_update_verified": True,
            "average_l2_weight_delta": avg_l2_delta,
            "weight_deltas_sample": {k: round(v, 6) for k, v in list(weight_deltas.items())[:5]}
        },
        "label_mapping": LABEL_MAPPING,
        "history": history
    }
    
    with open(output_dir / "model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    print(f"Model artifacts and manifest saved to: {output_dir}", flush=True)
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--encoder_lr", type=float, default=3e-5)
    parser.add_argument("--classifier_lr", type=float, default=8e-4)
    parser.add_argument("--samples", type=int, default=800)
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "artifacts" / "dataset_audit"
    out_dir = project_root / "artifacts" / "finetuned_sbert"
    
    train_sbert(
        data_dir=data_dir,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        classifier_lr=args.classifier_lr,
        max_train_samples=args.samples
    )
