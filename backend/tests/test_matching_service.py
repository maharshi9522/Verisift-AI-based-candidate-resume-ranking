import io
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token, hash_password
from app.main import app
from app.models.candidate import Candidate
from app.models.job import Job
from app.models.user import User
from app.services.matching_service import MatchingService

client = TestClient(app)

def test_relevance_score_formula_bounds():
    # Test extremes and midpoints
    # 1. Pure No Fit
    p_no = np.array([1.0, 0.0, 0.0])
    assert MatchingService.calculate_relevance_score(p_no) == 0.0

    # 2. Pure Good Fit
    p_good = np.array([0.0, 0.0, 1.0])
    assert MatchingService.calculate_relevance_score(p_good) == 100.0

    # 3. Pure Potential Fit
    p_pot = np.array([0.0, 1.0, 0.0])
    assert MatchingService.calculate_relevance_score(p_pot) == 50.0

    # 4. Uniform distribution
    p_uni = np.array([1/3, 1/3, 1/3])
    expected = 100.0 * (0.5 * (1/3) + 1.0 * (1/3))  # 50.0
    assert abs(MatchingService.calculate_relevance_score(p_uni) - expected) < 1e-4

def test_pairwise_feature_construction():
    r_emb = np.ones(384, dtype=np.float32)
    j_emb = np.ones(384, dtype=np.float32) * 2.0
    feat = MatchingService.compute_pairwise_features(r_emb, j_emb)
    # Dimension must be 384 * 4 = 1536
    assert feat.shape == (1536,)
    # r is ones, j is 2s, diff is 1s, prod is 2s
    assert np.allclose(feat[:384], 1.0)
    assert np.allclose(feat[384:768], 2.0)
    assert np.allclose(feat[768:1152], 1.0)
    assert np.allclose(feat[1152:1536], 2.0)

from app.db.base import Base
from tests.test_auth import client, test_engine, TestingSessionLocal

# Ensure test database schema exists
Base.metadata.create_all(bind=test_engine)

def test_rank_job_candidates_api():
    # Setup test user and job in test database
    db = TestingSessionLocal()
    try:
        # Create test recruiter
        test_email = "matching_recruiter@verisift.com"
        user = db.query(User).filter(User.email == test_email).first()
        if not user:
            user = User(
                email=test_email,
                password_hash=hash_password("Password123!"),
                full_name="Matching Recruiter"
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        # Create test job
        job = Job(
            recruiter_id=user.id,
            title="Senior PyTorch Machine Learning Engineer",
            department="Engineering",
            description="Looking for an experienced ML engineer with deep learning, PyTorch, Transformers, NLP, and model evaluation experience."
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        import uuid
        u1, u2 = uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
        # Create two candidates with extracted text
        cand1 = Candidate(
            job_id=job.id,
            recruiter_id=user.id,
            original_filename="strong_ml_engineer.pdf",
            stored_filename=f"strong_ml_{u1}.pdf",
            file_type="pdf",
            file_size=1024,
            file_hash=f"hash_ml_{u1}",
            extracted_text="Senior Machine Learning Engineer with 6 years experience in Python, PyTorch, Transformers, BERT, LLMs, and model ranking.",
            processing_status="completed"
        )
        cand2 = Candidate(
            job_id=job.id,
            recruiter_id=user.id,
            original_filename="accountant.pdf",
            stored_filename=f"accountant_{u2}.pdf",
            file_type="pdf",
            file_size=1024,
            file_hash=f"hash_acc_{u2}",
            extracted_text="Certified Accountant with experience in QuickBooks, payroll, financial auditing, tax preparation, and general ledger accounts.",
            processing_status="completed"
        )
        db.add_all([cand1, cand2])
        db.commit()

        # Auth token
        token = create_access_token({"sub": str(user.id)})
        headers = {"Authorization": f"Bearer {token}"}

        # Call rank endpoint
        res = client.post(f"/api/jobs/{job.id}/rank", headers=headers)
        assert res.status_code == 200, res.text
        data = res.json()

        assert data["job_id"] == job.id
        assert data["total_candidates_ranked"] == 2
        ranked = data["ranked_candidates"]
        assert len(ranked) == 2

        # Verify ranking order (descending by score)
        assert ranked[0]["relevance_score"] >= ranked[1]["relevance_score"]
        assert ranked[0]["ranking_position"] == 1
        assert ranked[1]["ranking_position"] == 2

        # Strong ML candidate must have higher score than accountant
        assert ranked[0]["original_filename"] == "strong_ml_engineer.pdf"
        assert ranked[1]["original_filename"] == "accountant.pdf"
        assert ranked[0]["relevance_score"] > ranked[1]["relevance_score"]

        # Verify scores are bounded between 0 and 100
        assert 0.0 <= ranked[0]["relevance_score"] <= 100.0
        assert 0.0 <= ranked[1]["relevance_score"] <= 100.0

        # Verify persistence in database
        db.refresh(cand1)
        db.refresh(cand2)
        assert cand1.relevance_score is not None
        assert cand1.ranking_position == 1
        assert cand1.fit_category in ["No Fit", "Potential Fit", "Good Fit"]
        assert cand2.ranking_position == 2

    finally:
        db.close()

def test_rank_empty_job_candidates():
    db = TestingSessionLocal()
    try:
        test_email = "empty_job_recruiter@verisift.com"
        user = db.query(User).filter(User.email == test_email).first()
        if not user:
            user = User(
                email=test_email,
                password_hash=hash_password("Password123!"),
                full_name="Empty Job Recruiter"
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        job = Job(
            recruiter_id=user.id,
            title="Empty Job Title",
            department="Product",
            description="Role with no candidates yet."
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        token = create_access_token({"sub": str(user.id)})
        headers = {"Authorization": f"Bearer {token}"}

        res = client.post(f"/api/jobs/{job.id}/rank", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["job_id"] == job.id
        assert data["total_candidates_ranked"] == 0
        assert data["ranked_candidates"] == []
    finally:
        db.close()

def test_rank_nonexistent_job():
    db = TestingSessionLocal()
    try:
        test_email = "nonexistent_recruiter@verisift.com"
        user = db.query(User).filter(User.email == test_email).first()
        if not user:
            user = User(
                email=test_email,
                password_hash=hash_password("Password123!"),
                full_name="Nonexistent Recruiter"
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        token = create_access_token({"sub": str(user.id)})
        headers = {"Authorization": f"Bearer {token}"}

        res = client.post("/api/jobs/999999/rank", headers=headers)
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()
    finally:
        db.close()

