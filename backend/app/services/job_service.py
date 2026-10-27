from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_
from fastapi import HTTPException, status
from app.models.job import Job
from app.schemas.job import JobCreate, JobUpdate

def utc_now():
    return datetime.now(timezone.utc)

class JobService:
    @staticmethod
    def create_job(db: Session, recruiter_id: int, job_in: JobCreate) -> Job:
        job = Job(
            recruiter_id=recruiter_id,
            title=job_in.title,
            department=job_in.department,
            location=job_in.location,
            employment_type=job_in.employment_type,
            workplace_type=job_in.workplace_type,
            description=job_in.description,
            required_skills=job_in.required_skills,
            preferred_skills=job_in.preferred_skills,
            minimum_experience=job_in.minimum_experience,
            status=job_in.status,
            created_at=utc_now(),
            updated_at=utc_now()
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def get_recruiter_jobs(
        db: Session,
        recruiter_id: int,
        status_filter: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> List[Job]:
        query = db.query(Job).filter(Job.recruiter_id == recruiter_id)

        if status_filter:
            norm_status = status_filter.strip().lower()
            if norm_status != "all":
                query = query.filter(Job.status == norm_status)

        if search_query:
            term = f"%{search_query.strip()}%"
            query = query.filter(
                or_(
                    Job.title.ilike(term),
                    Job.department.ilike(term),
                    Job.location.ilike(term)
                )
            )

        return query.order_by(Job.created_at.desc()).all()

    @staticmethod
    def get_job_by_id(db: Session, recruiter_id: int, job_id: int) -> Job:
        job = db.query(Job).filter(Job.id == job_id, Job.recruiter_id == recruiter_id).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job role not found or access denied."
            )
        return job

    @staticmethod
    def update_job(db: Session, recruiter_id: int, job_id: int, job_in: JobUpdate) -> Job:
        job = JobService.get_job_by_id(db, recruiter_id, job_id)

        update_data = job_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(job, field, value)

        job.updated_at = utc_now()
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def delete_job(db: Session, recruiter_id: int, job_id: int) -> int:
        job = JobService.get_job_by_id(db, recruiter_id, job_id)
        db.delete(job)
        db.commit()
        return job_id

    @staticmethod
    def archive_job(db: Session, recruiter_id: int, job_id: int) -> Job:
        job = JobService.get_job_by_id(db, recruiter_id, job_id)
        job.status = "archived"
        job.updated_at = utc_now()
        db.commit()
        db.refresh(job)
        return job
