import pytest
import os
import tempfile
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.models.user import User

# Setup isolated test database in a temporary file to also test persistence
TEST_DB_FILE = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
TEST_DB_FILE.close()
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_FILE.name}"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    if os.path.exists(TEST_DB_FILE.name):
        try:
            os.remove(TEST_DB_FILE.name)
        except OSError:
            pass

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Verisift"

def test_successful_registration_and_hash_storage():
    payload = {
        "full_name": "Elena Rostova",
        "email": "elena.recruiter@verisift.io",
        "password": "SecurePassword123!",
        "confirm_password": "SecurePassword123!"
    }
    response = client.post("/api/auth/signup", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "elena.recruiter@verisift.io"
    assert data["user"]["full_name"] == "Elena Rostova"
    assert "password" not in data["user"]
    assert "password_hash" not in data["user"]

    # Verify directly in SQLite that password is NOT stored in plaintext
    db = TestingSessionLocal()
    user = db.query(User).filter(User.email == "elena.recruiter@verisift.io").first()
    assert user is not None
    assert user.password_hash != "SecurePassword123!"
    assert user.password_hash.startswith("$2b$") or user.password_hash.startswith("$2a$")
    db.close()

def test_duplicate_email_registration_rejection():
    payload = {
        "full_name": "Duplicate Elena",
        "email": "ELENA.RECRUITER@VERISIFT.IO",  # uppercase to test normalization
        "password": "SecurePassword123!",
        "confirm_password": "SecurePassword123!"
    }
    response = client.post("/api/auth/signup", json=payload)
    assert response.status_code == 409
    data = response.json()
    assert "already exists" in data["detail"]

def test_invalid_email_format():
    payload = {
        "full_name": "Test User",
        "email": "not-an-email-address",
        "password": "SecurePassword123!"
    }
    response = client.post("/api/auth/signup", json=payload)
    assert response.status_code == 422

def test_successful_login():
    payload = {
        "email": "elena.recruiter@verisift.io",
        "password": "SecurePassword123!"
    }
    response = client.post("/api/auth/login", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "elena.recruiter@verisift.io"

def test_invalid_password_rejection():
    payload = {
        "email": "elena.recruiter@verisift.io",
        "password": "WrongPassword999!"
    }
    response = client.post("/api/auth/login", json=payload)
    assert response.status_code == 401
    data = response.json()
    assert data["detail"] == "Invalid email or password."

def test_nonexistent_email_rejection():
    payload = {
        "email": "nobody@verisift.io",
        "password": "SomePassword123!"
    }
    response = client.post("/api/auth/login", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."

def test_protected_endpoint_without_auth():
    response = client.get("/api/auth/me")
    assert response.status_code == 401

def test_current_user_retrieval_and_logout():
    # Login to obtain a fresh token
    login_res = client.post("/api/auth/login", json={
        "email": "elena.recruiter@verisift.io",
        "password": "SecurePassword123!"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Access /api/auth/me
    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "elena.recruiter@verisift.io"
    assert me_res.json()["full_name"] == "Elena Rostova"

    # Logout
    logout_res = client.post("/api/auth/logout", headers=headers)
    assert logout_res.status_code == 200
    assert logout_res.json()["success"] is True

    # After logout, accessing /api/auth/me with same token must fail
    revoked_res = client.get("/api/auth/me", headers=headers)
    assert revoked_res.status_code == 401

def test_database_persistence_across_connections():
    # Write with one session, read with another
    db1 = TestingSessionLocal()
    new_user = User(
        full_name="Persistence Test",
        email="persist@verisift.io",
        password_hash="fakehash123"
    )
    db1.add(new_user)
    db1.commit()
    user_id = new_user.id
    db1.close()

    db2 = TestingSessionLocal()
    retrieved = db2.query(User).filter(User.id == user_id).first()
    assert retrieved is not None
    assert retrieved.email == "persist@verisift.io"
    db2.close()
