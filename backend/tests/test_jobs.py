import pytest
from app.models.user import User
from app.models.job import Job
from app.core.security import hash_password, create_access_token
from tests.test_auth import client, TestingSessionLocal

@pytest.fixture
def auth_headers_user1():
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == "recruiter1@verisift.io").first()
    if not user:
        user = User(
            full_name="Sarah Jenkins",
            email="recruiter1@verisift.io",
            password_hash=hash_password("Password123!"),
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    token = create_access_token({"sub": str(user.id), "email": user.email})
    db.close()
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def auth_headers_user2():
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == "recruiter2@verisift.io").first()
    if not user:
        user = User(
            full_name="David Chen",
            email="recruiter2@verisift.io",
            password_hash=hash_password("Password123!"),
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    token = create_access_token({"sub": str(user.id), "email": user.email})
    db.close()
    return {"Authorization": f"Bearer {token}"}

def test_unauthenticated_access_blocked():
    """Unauthenticated requests must be rejected with 401."""
    res_list = client.get("/api/jobs")
    assert res_list.status_code == 401

    res_create = client.post("/api/jobs", json={"title": "Test Job", "description": "Short desc here"})
    assert res_create.status_code == 401

    res_get = client.get("/api/jobs/1")
    assert res_get.status_code == 401

    res_update = client.patch("/api/jobs/1", json={"title": "Updated"})
    assert res_update.status_code == 401

    res_del = client.delete("/api/jobs/1")
    assert res_del.status_code == 401

    res_arch = client.post("/api/jobs/1/archive")
    assert res_arch.status_code == 401

def test_create_job_success(auth_headers_user1):
    payload = {
        "title": "Senior Machine Learning Engineer",
        "department": "Engineering",
        "location": "San Francisco, CA",
        "employment_type": "Full-time",
        "workplace_type": "Hybrid",
        "description": "Design and productionize embedding models for candidate scoring and retrieval systems.",
        "required_skills": ["Python", "PyTorch", "FastAPI"],
        "preferred_skills": ["HuggingFace", "Docker", "Vector DBs"],
        "minimum_experience": 4.5,
        "status": "draft"
    }
    response = client.post("/api/jobs", json=payload, headers=auth_headers_user1)
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == payload["title"]
    assert data["department"] == "Engineering"
    assert data["location"] == "San Francisco, CA"
    assert data["employment_type"] == "Full-time"
    assert data["workplace_type"] == "Hybrid"
    assert data["status"] == "draft"
    assert data["required_skills"] == ["Python", "PyTorch", "FastAPI"]
    assert data["preferred_skills"] == ["HuggingFace", "Docker", "Vector DBs"]
    assert data["minimum_experience"] == 4.5
    assert "id" in data
    assert "recruiter_id" in data
    assert "created_at" in data

def test_validation_errors(auth_headers_user1):
    # Blank title
    res1 = client.post("/api/jobs", json={
        "title": "   ",
        "description": "Valid description with sufficient length here."
    }, headers=auth_headers_user1)
    assert res1.status_code == 422

    # Description too short
    res2 = client.post("/api/jobs", json={
        "title": "Engineering Manager",
        "description": "Short"
    }, headers=auth_headers_user1)
    assert res2.status_code == 422

    # Negative minimum experience
    res3 = client.post("/api/jobs", json={
        "title": "Engineering Manager",
        "description": "A very detailed description of this leadership role.",
        "minimum_experience": -2
    }, headers=auth_headers_user1)
    assert res3.status_code == 422

    # Invalid employment type
    res4 = client.post("/api/jobs", json={
        "title": "Engineering Manager",
        "description": "A very detailed description of this leadership role.",
        "employment_type": "Casual-Weekend"
    }, headers=auth_headers_user1)
    assert res4.status_code == 422

def test_list_and_search_jobs(auth_headers_user1):
    # Create active role
    client.post("/api/jobs", json={
        "title": "Full Stack Engineer",
        "department": "Product",
        "location": "New York, NY",
        "employment_type": "Full-time",
        "workplace_type": "Remote",
        "description": "Building modern recruiter-first interfaces with React and Python.",
        "required_skills": ["TypeScript", "React"],
        "status": "active"
    }, headers=auth_headers_user1)

    # List all jobs
    res = client.get("/api/jobs", headers=auth_headers_user1)
    assert res.status_code == 200
    jobs = res.json()
    assert len(jobs) >= 2

    # Filter by status: active
    res_active = client.get("/api/jobs?status=active", headers=auth_headers_user1)
    assert res_active.status_code == 200
    active_jobs = res_active.json()
    assert all(j["status"] == "active" for j in active_jobs)
    assert any(j["title"] == "Full Stack Engineer" for j in active_jobs)

    # Filter by search
    res_search = client.get("/api/jobs?search=Machine Learning", headers=auth_headers_user1)
    assert res_search.status_code == 200
    searched = res_search.json()
    assert len(searched) >= 1
    assert any("Machine Learning" in j["title"] for j in searched)

def test_ownership_isolation(auth_headers_user1, auth_headers_user2):
    """Confirm one recruiter cannot view, update, archive, or delete another recruiter's job."""
    # User 1 creates a private job
    res_create = client.post("/api/jobs", json={
        "title": "Confidential VP of AI",
        "department": "Executive",
        "description": "Leading organizational artificial intelligence strategy and roadmap.",
        "status": "draft"
    }, headers=auth_headers_user1)
    job_id = res_create.json()["id"]

    # User 2 lists jobs - should NOT contain User 1's job
    res_u2_list = client.get("/api/jobs", headers=auth_headers_user2)
    assert res_u2_list.status_code == 200
    u2_job_ids = [j["id"] for j in res_u2_list.json()]
    assert job_id not in u2_job_ids

    # User 2 tries to GET job directly - 404
    res_u2_get = client.get(f"/api/jobs/{job_id}", headers=auth_headers_user2)
    assert res_u2_get.status_code == 404

    # User 2 tries to PATCH job - 404
    res_u2_patch = client.patch(f"/api/jobs/{job_id}", json={"title": "Hacked Title"}, headers=auth_headers_user2)
    assert res_u2_patch.status_code == 404

    # User 2 tries to archive job - 404
    res_u2_arch = client.post(f"/api/jobs/{job_id}/archive", headers=auth_headers_user2)
    assert res_u2_arch.status_code == 404

    # User 2 tries to DELETE job - 404
    res_u2_del = client.delete(f"/api/jobs/{job_id}", headers=auth_headers_user2)
    assert res_u2_del.status_code == 404

    # Verify job is still unchanged for User 1
    res_verify = client.get(f"/api/jobs/{job_id}", headers=auth_headers_user1)
    assert res_verify.status_code == 200
    assert res_verify.json()["title"] == "Confidential VP of AI"

def test_update_archive_and_delete_job(auth_headers_user1):
    # Create role to test lifecycle
    res_create = client.post("/api/jobs", json={
        "title": "Product Designer",
        "department": "Design",
        "description": "Crafting clean, accessible, recruiter-first user interfaces.",
        "status": "draft"
    }, headers=auth_headers_user1)
    job_id = res_create.json()["id"]

    # Update role
    res_update = client.patch(f"/api/jobs/{job_id}", json={
        "title": "Lead Product Designer",
        "status": "active",
        "minimum_experience": 5.0
    }, headers=auth_headers_user1)
    assert res_update.status_code == 200
    assert res_update.json()["title"] == "Lead Product Designer"
    assert res_update.json()["status"] == "active"
    assert res_update.json()["minimum_experience"] == 5.0

    # Archive role
    res_arch = client.post(f"/api/jobs/{job_id}/archive", headers=auth_headers_user1)
    assert res_arch.status_code == 200
    assert res_arch.json()["status"] == "archived"

    # Delete role
    res_del = client.delete(f"/api/jobs/{job_id}", headers=auth_headers_user1)
    assert res_del.status_code == 200
    assert res_del.json()["id"] == job_id

    # Verify deletion
    res_get_del = client.get(f"/api/jobs/{job_id}", headers=auth_headers_user1)
    assert res_get_del.status_code == 404
