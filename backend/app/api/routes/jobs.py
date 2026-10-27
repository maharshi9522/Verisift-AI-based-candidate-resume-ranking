from typing import List, Optional
from fastapi import APIRouter, Depends, status, Query
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.user import User
from app.api.routes.auth import get_current_user
from app.schemas.job import JobCreate, JobUpdate, JobResponse, DeleteJobResponse
from app.services.job_service import JobService

router = APIRouter(prefix="/jobs", tags=["Jobs"])

@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
def create_job(
    job_in: JobCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Creates a new job role belonging to the authenticated recruiter."""
    return JobService.create_job(db=db, recruiter_id=current_user.id, job_in=job_in)

@router.get("", response_model=List[JobResponse])
def list_jobs(
    status: Optional[str] = Query(default=None, description="Filter by status (draft, active, archived)"),
    search: Optional[str] = Query(default=None, description="Search by title, department, or location"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lists all job roles created by the authenticated recruiter."""
    return JobService.get_recruiter_jobs(
        db=db,
        recruiter_id=current_user.id,
        status_filter=status,
        search_query=search
    )

@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves a specific job role owned by the authenticated recruiter."""
    return JobService.get_job_by_id(db=db, recruiter_id=current_user.id, job_id=job_id)

@router.patch("/{job_id}", response_model=JobResponse)
def update_job(
    job_id: int,
    job_in: JobUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Updates a job role owned by the authenticated recruiter."""
    return JobService.update_job(
        db=db,
        recruiter_id=current_user.id,
        job_id=job_id,
        job_in=job_in
    )

@router.delete("/{job_id}", response_model=DeleteJobResponse)
def delete_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deletes a job role owned by the authenticated recruiter."""
    deleted_id = JobService.delete_job(db=db, recruiter_id=current_user.id, job_id=job_id)
    return DeleteJobResponse(message="Job role deleted successfully", id=deleted_id)

@router.post("/{job_id}/archive", response_model=JobResponse)
def archive_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Archives a job role owned by the authenticated recruiter."""
    return JobService.archive_job(db=db, recruiter_id=current_user.id, job_id=job_id)
