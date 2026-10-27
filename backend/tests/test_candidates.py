import io
import os
import pytest
import pymupdf
import docx
from app.models.user import User
from app.models.job import Job
from app.models.candidate import Candidate
from app.core.security import hash_password, create_access_token
from app.core.config import settings
from tests.test_auth import client, TestingSessionLocal

def create_sample_pdf(text: str = "Jane Doe\nSenior Machine Learning Engineer\nExtensive experience with PyTorch and NLP.") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def create_image_only_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    # Draw a shape without any text
    page.draw_rect(pymupdf.Rect(50, 50, 200, 200), color=(0.1, 0.5, 0.8))
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def create_sample_docx(name: str = "Alex Mercer", summary: str = "Full Stack Engineer specializing in React and Python.") -> bytes:
    doc = docx.Document()
    doc.add_heading(name, 0)
    doc.add_paragraph(summary)
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Core Competencies"
    table.rows[0].cells[1].text = "Python, TypeScript, SQL, Docker"
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()

@pytest.fixture
def recruiter1():
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
    return {"id": user.id, "headers": {"Authorization": f"Bearer {token}"}}

@pytest.fixture
def recruiter2():
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
    return {"id": user.id, "headers": {"Authorization": f"Bearer {token}"}}

@pytest.fixture
def job_r1(recruiter1):
    db = TestingSessionLocal()
    job = Job(
        recruiter_id=recruiter1["id"],
        title="ML Infrastructure Lead",
        department="Engineering",
        description="Lead large-scale distributed inference infrastructure and candidate retrieval.",
        status="active"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    job_id = job.id
    db.close()
    return job_id

def test_unauthenticated_candidate_endpoints_rejected(job_r1):
    """Unauthenticated requests must be rejected with 401."""
    pdf_bytes = create_sample_pdf()

    r1 = client.post(f"/api/jobs/{job_r1}/candidates/upload", files=[("files", ("resume.pdf", pdf_bytes, "application/pdf"))])
    assert r1.status_code == 401

    r2 = client.get(f"/api/jobs/{job_r1}/candidates")
    assert r2.status_code == 401

    r3 = client.get("/api/candidates/1")
    assert r3.status_code == 401

    r4 = client.delete("/api/candidates/1")
    assert r4.status_code == 401

    r5 = client.post("/api/candidates/1/retry-extraction")
    assert r5.status_code == 401

def test_successful_pdf_upload_and_extraction(recruiter1, job_r1):
    """PDF upload should succeed and extract text page by page."""
    pdf_content = create_sample_pdf("Dr. Jane Doe\nSenior AI Researcher with 8 years in NLP and Transformers.")
    files = [("files", ("jane_doe_resume.pdf", pdf_content, "application/pdf"))]

    response = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=files,
        headers=recruiter1["headers"]
    )
    assert response.status_code == 201
    data = response.json()
    assert data["total_uploaded"] == 1
    assert data["total_duplicates"] == 0
    assert data["total_errors"] == 0

    uploaded = data["uploaded"][0]
    assert uploaded["original_filename"] == "jane_doe_resume.pdf"
    assert uploaded["file_type"] == "pdf"
    assert uploaded["processing_status"] == "completed"
    assert uploaded["job_id"] == job_r1

    # Verify candidate detail includes extracted text
    candidate_id = uploaded["id"]
    detail_res = client.get(f"/api/candidates/{candidate_id}", headers=recruiter1["headers"])
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert "Jane Doe" in detail_data["extracted_text"]
    assert "Senior AI Researcher" in detail_data["extracted_text"]

def test_successful_docx_upload_and_extraction(recruiter1, job_r1):
    """DOCX upload should succeed and extract body paragraphs and tables."""
    docx_content = create_sample_docx("Marcus Brody", "Distinguished Cloud Architect with Kubernetes expertise.")
    files = [("files", ("marcus_brody.docx", docx_content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))]

    response = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=files,
        headers=recruiter1["headers"]
    )
    assert response.status_code == 201
    data = response.json()
    assert data["total_uploaded"] == 1
    uploaded = data["uploaded"][0]
    assert uploaded["processing_status"] == "completed"

    candidate_id = uploaded["id"]
    detail_res = client.get(f"/api/candidates/{candidate_id}", headers=recruiter1["headers"])
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert "Marcus Brody" in detail_data["extracted_text"]
    assert "Core Competencies" in detail_data["extracted_text"]

def test_multiple_file_batch_upload(recruiter1, job_r1):
    """Multiple files uploaded simultaneously should all be processed."""
    pdf1 = create_sample_pdf("Candidate Alpha\nBackend Engineer with Go and PostgreSQL experience.")
    pdf2 = create_sample_pdf("Candidate Beta\nFrontend Architect with React, Vite, and CSS systems.")
    docx1 = create_sample_docx("Candidate Gamma", "Data Engineer with Snowflake and DBT background.")

    files = [
        ("files", ("alpha.pdf", pdf1, "application/pdf")),
        ("files", ("beta.pdf", pdf2, "application/pdf")),
        ("files", ("gamma.docx", docx1, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
    ]

    response = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=files,
        headers=recruiter1["headers"]
    )
    assert response.status_code == 201
    data = response.json()
    assert data["total_uploaded"] == 3
    assert len(data["uploaded"]) == 3
    assert all(item["processing_status"] == "completed" for item in data["uploaded"])

def test_unsupported_file_type_rejected(recruiter1, job_r1):
    """Files with unsupported extensions (like .txt or .exe) must be reported in errors list."""
    txt_content = b"Plain text resume content here."
    files = [("files", ("candidate.txt", txt_content, "text/plain"))]

    response = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=files,
        headers=recruiter1["headers"]
    )
    assert response.status_code == 201
    data = response.json()
    assert data["total_uploaded"] == 0
    assert data["total_errors"] == 1
    assert "Unsupported file type" in data["errors"][0]["reason"]

def test_empty_and_malformed_file_rejected(recruiter1, job_r1):
    """Empty files (0 bytes) or corrupted headers must be flagged as errors without crashing batch."""
    empty_bytes = b""
    malformed_pdf = b"NOT_A_REAL_PDF_HEADER_1234567890"

    files = [
        ("files", ("empty.pdf", empty_bytes, "application/pdf")),
        ("files", ("corrupt.pdf", malformed_pdf, "application/pdf")),
    ]

    response = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=files,
        headers=recruiter1["headers"]
    )
    assert response.status_code == 201
    data = response.json()
    assert data["total_uploaded"] == 0
    assert data["total_errors"] == 2
    assert any("empty" in err["reason"].lower() for err in data["errors"])
    assert any("malformed" in err["reason"].lower() for err in data["errors"])

def test_oversized_file_rejected(recruiter1, job_r1, monkeypatch):
    """Files exceeding maximum allowed size (e.g. simulated > 10MB) must be rejected."""
    # Temporarily monkeypatch MAX_FILE_SIZE_BYTES to 1000 bytes to test limit without allocating 10MB
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_BYTES", 500)
    oversized_content = b"%PDF-" + b"A" * 1000

    files = [("files", ("oversized.pdf", oversized_content, "application/pdf"))]
    response = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=files,
        headers=recruiter1["headers"]
    )
    assert response.status_code == 201
    data = response.json()
    assert data["total_uploaded"] == 0
    assert data["total_errors"] == 1
    assert "exceeds maximum allowed size" in data["errors"][0]["reason"]

def test_duplicate_upload_detection(recruiter1, job_r1):
    """Uploading the exact same file content twice for the same job must flag duplicate."""
    pdf_content = create_sample_pdf("Unique Candidate Content for Deduplication Test 12345")

    # First upload
    res1 = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=[("files", ("first_upload.pdf", pdf_content, "application/pdf"))],
        headers=recruiter1["headers"]
    )
    assert res1.status_code == 201
    assert res1.json()["total_uploaded"] == 1

    # Second upload with identical content (even with different filename)
    res2 = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=[("files", ("duplicate_copy.pdf", pdf_content, "application/pdf"))],
        headers=recruiter1["headers"]
    )
    assert res2.status_code == 201
    data2 = res2.json()
    assert data2["total_uploaded"] == 0
    assert data2["total_duplicates"] == 1
    assert "already uploaded" in data2["duplicates"][0]["message"].lower()

def test_image_only_pdf_extraction_failure(recruiter1, job_r1):
    """A PDF with no extractable text must be stored with processing_status='failed' and clear error."""
    image_pdf = create_image_only_pdf()
    files = [("files", ("scanned_invoice.pdf", image_pdf, "application/pdf"))]

    response = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=files,
        headers=recruiter1["headers"]
    )
    assert response.status_code == 201
    uploaded = response.json()["uploaded"][0]
    assert uploaded["processing_status"] == "failed"
    assert "image-only or scanned" in uploaded["error_message"]

def test_candidate_list_and_search_filter(recruiter1, job_r1):
    """Listing candidates must support filtering by status and search by filename."""
    pdf_alpha = create_sample_pdf("Alpha Candidate with Systems Engineering Skills")
    pdf_beta = create_sample_pdf("Beta Candidate with Data Science Skills")
    client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=[
            ("files", ("candidate_alpha.pdf", pdf_alpha, "application/pdf")),
            ("files", ("candidate_beta.pdf", pdf_beta, "application/pdf")),
        ],
        headers=recruiter1["headers"]
    )

    res = client.get(f"/api/jobs/{job_r1}/candidates", headers=recruiter1["headers"])
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert data["total"] == 2

    # Search for specific filename
    res_search = client.get(f"/api/jobs/{job_r1}/candidates?search=alpha", headers=recruiter1["headers"])
    assert res_search.status_code == 200
    search_items = res_search.json()["items"]
    assert len(search_items) == 1
    assert "alpha" in search_items[0]["original_filename"].lower()

    # Filter by status: completed
    res_status = client.get(f"/api/jobs/{job_r1}/candidates?status=completed", headers=recruiter1["headers"])
    assert res_status.status_code == 200
    status_items = res_status.json()["items"]
    assert all(item["processing_status"] == "completed" for item in status_items)

def test_ownership_isolation_candidates(recruiter1, recruiter2, job_r1):
    """Recruiter 2 cannot view, retry, or delete Recruiter 1's candidates."""
    # Recruiter 1 uploads candidate
    pdf_content = create_sample_pdf("Private Recruiter 1 Candidate Data")
    res_up = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=[("files", ("private_candidate.pdf", pdf_content, "application/pdf"))],
        headers=recruiter1["headers"]
    )
    candidate_id = res_up.json()["uploaded"][0]["id"]

    # Recruiter 2 tries to upload to Recruiter 1's job -> 404
    r2_upload = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=[("files", ("hacker.pdf", pdf_content, "application/pdf"))],
        headers=recruiter2["headers"]
    )
    assert r2_upload.status_code == 404

    # Recruiter 2 tries to list candidates for Recruiter 1's job -> 404
    r2_list = client.get(f"/api/jobs/{job_r1}/candidates", headers=recruiter2["headers"])
    assert r2_list.status_code == 404

    # Recruiter 2 tries to GET candidate details -> 404
    r2_detail = client.get(f"/api/candidates/{candidate_id}", headers=recruiter2["headers"])
    assert r2_detail.status_code == 404

    # Recruiter 2 tries to retry extraction -> 404
    r2_retry = client.post(f"/api/candidates/{candidate_id}/retry-extraction", headers=recruiter2["headers"])
    assert r2_retry.status_code == 404

    # Recruiter 2 tries to delete candidate -> 404
    r2_del = client.delete(f"/api/candidates/{candidate_id}", headers=recruiter2["headers"])
    assert r2_del.status_code == 404

def test_candidate_deletion_and_file_cleanup(recruiter1, job_r1):
    """Deleting a candidate must remove database record and remove stored physical file."""
    pdf_content = create_sample_pdf("Ephemeral Candidate to be deleted")
    res_up = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=[("files", ("to_delete.pdf", pdf_content, "application/pdf"))],
        headers=recruiter1["headers"]
    )
    candidate_id = res_up.json()["uploaded"][0]["id"]

    # Check physical file exists
    db = TestingSessionLocal()
    cand = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    stored_path = settings.UPLOAD_DIR / cand.stored_filename
    assert stored_path.exists()
    db.close()

    # Delete candidate
    del_res = client.delete(f"/api/candidates/{candidate_id}", headers=recruiter1["headers"])
    assert del_res.status_code == 200

    # Verify physical file was removed
    assert not stored_path.exists()

    # Subsequent GET returns 404
    check_res = client.get(f"/api/candidates/{candidate_id}", headers=recruiter1["headers"])
    assert check_res.status_code == 404

def test_retry_extraction(recruiter1, job_r1):
    """Retrying extraction updates status and text on eligible candidate."""
    pdf_content = create_sample_pdf("Valid Candidate Text for Retry Test")
    res_up = client.post(
        f"/api/jobs/{job_r1}/candidates/upload",
        files=[("files", ("retry_cand.pdf", pdf_content, "application/pdf"))],
        headers=recruiter1["headers"]
    )
    candidate_id = res_up.json()["uploaded"][0]["id"]

    # Manually simulate a transient failure in DB
    db = TestingSessionLocal()
    cand = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    cand.processing_status = "failed"
    cand.extracted_text = None
    cand.error_message = "Simulated transient failure."
    db.commit()
    db.close()

    # Call retry endpoint
    retry_res = client.post(f"/api/candidates/{candidate_id}/retry-extraction", headers=recruiter1["headers"])
    assert retry_res.status_code == 200
    data = retry_res.json()
    assert data["processing_status"] == "completed"
    assert "Valid Candidate Text" in data["extracted_text"]
    assert data["error_message"] is None
