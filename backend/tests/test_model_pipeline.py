import json
from pathlib import Path
import numpy as np
import pytest
import torch

from app.services.matching_service import MatchingService, PairwiseMLP, LABEL_NAMES

def test_checkpoint_dimension_regression():
    """
    CRITICAL REGRESSION TEST:
    Verifies that the deployed PyTorch checkpoint weight matrix dimensions
    match exactly the 1,536-dimensional feature vector produced by MatchingService.
    Prevents silent dimension mismatches (e.g. 1536 vs 1537).
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    checkpoint_path = project_root / "artifacts" / "semantic_ranker" / "semantic_ranker_mlp.pt"
    assert checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}"

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    input_weights = state_dict["net.0.weight"] # Linear(input_dim, 256)
    
    # Assert checkpoint input dimension is exactly 1536
    assert input_weights.shape == (256, 1536), f"Checkpoint input dim mismatch: expected (256, 1536), got {input_weights.shape}"

    # Verify MatchingService pairwise feature dimension is exactly 1536
    r_dummy = np.zeros(384, dtype=np.float32)
    j_dummy = np.zeros(384, dtype=np.float32)
    features = MatchingService.compute_pairwise_features(r_dummy, j_dummy)
    assert features.shape == (1536,), f"Feature dimension mismatch: expected (1536,), got {features.shape}"
    assert features.shape[0] == input_weights.shape[1]

def test_label_names_and_class_ordering():
    """Ensures class indices match [0: No Fit, 1: Potential Fit, 2: Good Fit]."""
    assert LABEL_NAMES[0] == "No Fit"
    assert LABEL_NAMES[1] == "Potential Fit"
    assert LABEL_NAMES[2] == "Good Fit"

def test_relevance_score_monotonicity_and_bounds():
    # Verify score bounds [0.0, 100.0]
    p_zero = np.array([1.0, 0.0, 0.0])
    p_full = np.array([0.0, 0.0, 1.0])
    p_mid = np.array([0.0, 1.0, 0.0])

    assert MatchingService.calculate_relevance_score(p_zero) == 0.0
    assert MatchingService.calculate_relevance_score(p_full) == 100.0
    assert MatchingService.calculate_relevance_score(p_mid) == 50.0

    # Ensure strictly monotonic ranking from Good Fit to No Fit
    p_higher = np.array([0.1, 0.2, 0.7])
    p_lower = np.array([0.7, 0.2, 0.1])
    assert MatchingService.calculate_relevance_score(p_higher) > MatchingService.calculate_relevance_score(p_lower)

def test_deterministic_ranking():
    """Identical embeddings must yield identical pairwise features and scores."""
    np.random.seed(42)
    r = np.random.randn(384).astype(np.float32)
    j = np.random.randn(384).astype(np.float32)
    r = r / np.linalg.norm(r)
    j = j / np.linalg.norm(j)

    feat1 = MatchingService.compute_pairwise_features(r, j)
    feat2 = MatchingService.compute_pairwise_features(r, j)
    assert np.array_equal(feat1, feat2)
