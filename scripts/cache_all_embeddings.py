"""
Verisift Embedding Caching Service
Precomputes and caches normalized embeddings for all unique resumes and job descriptions
in verisift_cleaned_all.csv using all-MiniLM-L6-v2.
Allows instant experimentation with any pairwise classification architecture.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

def cache_embeddings():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "artifacts" / "dataset_audit" / "splits" / "verisift_cleaned_all.csv"
    cache_dir = project_root / "artifacts" / "accuracy_experiments" / "embeddings_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    unique_resumes = df[["r_hash", "resume_text"]].drop_duplicates("r_hash").reset_index(drop=True)
    unique_jds = df[["j_hash", "job_description_text"]].drop_duplicates("j_hash").reset_index(drop=True)
    
    print(f"Unique resumes: {len(unique_resumes)}")
    print(f"Unique JDs: {len(unique_jds)}")
    print(f"Total unique documents: {len(unique_resumes) + len(unique_jds)}")
    
    torch.set_num_threads(14)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading encoder: {model_name}...")
    t0 = time.time()
    encoder = SentenceTransformer(model_name, device="cpu")
    print(f"Encoder loaded in {time.time() - t0:.2f}s")
    
    # 1. Standard MiniLM (256 tokens)
    print("Encoding resumes (max_seq_length=256)...")
    encoder.max_seq_length = 256
    t1 = time.time()
    resume_texts = unique_resumes["resume_text"].tolist()
    resume_embs_256 = encoder.encode(resume_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    print(f"Encoded {len(resume_texts)} resumes in {time.time() - t1:.2f}s. Shape: {resume_embs_256.shape}")
    
    print("Encoding job descriptions (max_seq_length=256)...")
    t2 = time.time()
    jd_texts = unique_jds["job_description_text"].tolist()
    jd_embs_256 = encoder.encode(jd_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    print(f"Encoded {len(jd_texts)} JDs in {time.time() - t2:.2f}s. Shape: {jd_embs_256.shape}")
    
    # Save 256-token embeddings
    cache_file_256 = cache_dir / "minilm_256_embeddings.npz"
    np.savez_compressed(
        cache_file_256,
        r_hashes=unique_resumes["r_hash"].values,
        r_embeddings=resume_embs_256,
        j_hashes=unique_jds["j_hash"].values,
        j_embeddings=jd_embs_256
    )
    print(f"Saved 256-token embeddings to {cache_file_256}")
    
    # 2. MiniLM with extended window (512 tokens - native max position embeddings)
    print("Encoding with max_seq_length=512 (native maximum for MiniLM)...")
    encoder.max_seq_length = 512
    t3 = time.time()
    resume_embs_512 = encoder.encode(resume_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    jd_embs_512 = encoder.encode(jd_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    print(f"Encoded 512-token embeddings in {time.time() - t3:.2f}s")
    
    cache_file_512 = cache_dir / "minilm_512_embeddings.npz"
    np.savez_compressed(
        cache_file_512,
        r_hashes=unique_resumes["r_hash"].values,
        r_embeddings=resume_embs_512,
        j_hashes=unique_jds["j_hash"].values,
        j_embeddings=jd_embs_512
    )
    print(f"Saved 512-token embeddings to {cache_file_512}")
    
    meta = {
        "model": model_name,
        "unique_resumes": len(unique_resumes),
        "unique_jds": len(unique_jds),
        "embedding_dim": 384,
        "cache_files": {
            "256_tokens": str(cache_file_256),
            "512_tokens": str(cache_file_512)
        }
    }
    with open(cache_dir / "cache_manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Embedding caching complete.")

if __name__ == "__main__":
    cache_embeddings()
