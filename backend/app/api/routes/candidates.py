from typing import List, Optional
from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.candidate import (
    BatchUploadResponse,
    CandidateDetailResponse,
    CandidateListResponse,
    DeleteCandidateResponse,
    RankJobResponse,
)
from app.services.candidate_service import CandidateService
from app.services.matching_service import MatchingService

router = APIRouter(tags=["Candidates"])

@router.post(
    "/jobs/{job_id}/rank",
    response_model=RankJobResponse,
    summary="Score and rank all candidates for a job role using deep semantic matching"
)
def rank_job_candidates(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Computes pair-aware deep semantic embeddings, predicts fit classes,
    assigns continuous ordinal relevance scores (0-100), and ranks candidates in descending order.
    """
    return MatchingService.rank_job_candidates(
        db=db,
        job_id=job_id,
        recruiter_id=current_user.id
    )

@router.post(
    "/jobs/{job_id}/candidates/upload",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload resumes for a job role"
)
async def upload_job_candidates(
    job_id: int,
    files: List[UploadFile] = File(..., description="PDF or DOCX resumes (max 10MB each)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Uploads multiple resume documents for a specific recruiter's job and extracts text."""
    return await CandidateService.upload_candidates(
        db=db,
        job_id=job_id,
        recruiter_id=current_user.id,
        files=files
    )

@router.get(
    "/jobs/{job_id}/candidates/export",
    summary="Download ranked candidate shortlist as PDF or Excel"
)
def export_job_candidates(
    job_id: int,
    limit: int = Query(default=100, ge=1, le=1000, description="Top N candidates to export (e.g. 100, 200, custom)"),
    format: str = Query(default="xlsx", pattern="^(xlsx|pdf)$", description="Export format: xlsx or pdf"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Downloads the authoritative ranked candidate shortlist for a job role in Excel (.xlsx) or PDF format.
    - Validates recruiter job ownership.
    - Preserves authoritative ranking position and relevance score.
    - Returns proper content type and Content-Disposition attachment.
    """
    return CandidateService.export_candidates_shortlist(
        db=db,
        job_id=job_id,
        recruiter_id=current_user.id,
        limit=limit,
        export_format=format.lower()
    )

@router.get(
    "/jobs/{job_id}/candidates",
    response_model=CandidateListResponse,
    summary="List candidates for a specific job role"
)
def list_job_candidates(
    job_id: int,
    search: Optional[str] = Query(default=None, description="Search by original filename"),
    status: Optional[str] = Query(default=None, description="Filter by status (processing, completed, failed, all)"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves paginated candidates uploaded against a specific recruiter's job."""
    return CandidateService.get_candidates_for_job(
        db=db,
        job_id=job_id,
        recruiter_id=current_user.id,
        search=search,
        status_filter=status,
        page=page,
        page_size=page_size
    )

@router.get(
    "/candidates",
    response_model=CandidateListResponse,
    summary="List all candidates for the authenticated recruiter"
)
def list_all_candidates(
    job_id: Optional[int] = Query(default=None, description="Filter candidates by job ID"),
    search: Optional[str] = Query(default=None, description="Search by original filename"),
    status: Optional[str] = Query(default=None, description="Filter by status (processing, completed, failed, all)"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves candidates across all jobs for the authenticated recruiter."""
    return CandidateService.get_all_candidates_for_recruiter(
        db=db,
        recruiter_id=current_user.id,
        job_id=job_id,
        search=search,
        status_filter=status,
        page=page,
        page_size=page_size
    )

@router.get(
    "/candidates/{candidate_id}",
    response_model=CandidateDetailResponse,
    summary="Retrieve candidate details including extracted resume text"
)
def get_candidate(
    candidate_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves full candidate details, metadata, and extracted text."""
    return CandidateService.get_candidate_details(
        db=db,
        candidate_id=candidate_id,
        recruiter_id=current_user.id
    )

@router.delete(
    "/candidates/{candidate_id}",
    response_model=DeleteCandidateResponse,
    summary="Delete a candidate record and clean up stored file"
)
def delete_candidate(
    candidate_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Permanently deletes candidate record and cleans up the stored physical document."""
    deleted_id = CandidateService.delete_candidate(
        db=db,
        candidate_id=candidate_id,
        recruiter_id=current_user.id
    )
    return DeleteCandidateResponse(message="Candidate record deleted successfully", id=deleted_id)

@router.post(
    "/candidates/{candidate_id}/retry-extraction",
    response_model=CandidateDetailResponse,
    summary="Retry text extraction for an eligible candidate"
)
def retry_candidate_extraction(
    candidate_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retries text extraction on the stored document for a failed candidate."""
    return CandidateService.retry_extraction(
        db=db,
        candidate_id=candidate_id,
        recruiter_id=current_user.id
    )
