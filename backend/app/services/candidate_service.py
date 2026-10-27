import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import re
from fastapi import HTTPException, UploadFile, status, Response
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.core.config import settings
from app.models.candidate import Candidate
from app.models.job import Job
from app.schemas.candidate import (
    BatchUploadResponse,
    CandidateResponse,
    CandidateDetailResponse,
    CandidateListResponse,
    DuplicateUploadItem,
    UploadErrorItem,
)
import json
from app.services.text_extraction_service import TextExtractionService
from app.services.resume_parser import ResumeParser
from app.services.export_service import ExportService

def utc_now():
    return datetime.now(timezone.utc)

# File signatures for validation
PDF_MAGIC_BYTES = b"%PDF-"
DOCX_MAGIC_BYTES = b"PK\x03\x04"

class CandidateService:
    @staticmethod
    def _verify_job_ownership(db: Session, job_id: int, recruiter_id: int) -> Job:
        """Verifies that the job exists and belongs to the authenticated recruiter."""
        job = db.query(Job).filter(Job.id == job_id, Job.recruiter_id == recruiter_id).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job role not found or access denied."
            )
        return job

    @staticmethod
    def validate_file_content(filename: str, content: bytes) -> Tuple[bool, str, str]:
        """
        Validates extension, size, and header signature.
        Returns: (is_valid, file_type, error_reason)
        """
        if not filename or not filename.strip():
            return False, "", "Filename is missing or empty."

        clean_name = filename.strip()
        ext = os.path.splitext(clean_name)[1].lower()

        if ext not in settings.ALLOWED_EXTENSIONS:
            return False, "", f"Unsupported file type '{ext}'. Only PDF and DOCX files are allowed."

        file_size = len(content)
        if file_size == 0:
            return False, "", "File is empty (0 bytes)."

        if file_size > settings.MAX_FILE_SIZE_BYTES:
            max_mb = settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)
            return False, "", f"File exceeds maximum allowed size of {max_mb} MB."

        # File signature / magic bytes validation
        file_type = ext.lstrip(".")
        if ext == ".pdf":
            if not content.startswith(PDF_MAGIC_BYTES):
                return False, "", "Malformed PDF file: missing standard PDF header signature."
        elif ext == ".docx":
            if not content.startswith(DOCX_MAGIC_BYTES):
                return False, "", "Malformed DOCX file: missing valid OpenXML document container signature."

        return True, file_type, ""

    @classmethod
    async def upload_candidates(
        cls,
        db: Session,
        job_id: int,
        recruiter_id: int,
        files: List[UploadFile]
    ) -> BatchUploadResponse:
        """Processes multiple resume uploads for a recruiter's job with deduplication and text extraction."""
        job = cls._verify_job_ownership(db, job_id, recruiter_id)

        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided in upload request."
            )

        if len(files) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size limit exceeded. Maximum {settings.MAX_BATCH_SIZE} files allowed per request."
            )

        uploaded_records: List[CandidateResponse] = []
        duplicate_records: List[DuplicateUploadItem] = []
        error_records: List[UploadErrorItem] = []

        for upload_file in files:
            filename = upload_file.filename or "unknown_file"

            try:
                # Read content safely
                content = await upload_file.read()

                # Validate content and signature
                is_valid, file_type, reason = cls.validate_file_content(filename, content)
                if not is_valid:
                    error_records.append(UploadErrorItem(filename=filename, reason=reason))
                    continue

                # Compute SHA-256 hash
                file_hash = hashlib.sha256(content).hexdigest()

                # Deduplication check: scope duplicate check to this job and recruiter
                existing = db.query(Candidate).filter(
                    Candidate.job_id == job_id,
                    Candidate.recruiter_id == recruiter_id,
                    Candidate.file_hash == file_hash
                ).first()

                if existing:
                    duplicate_records.append(DuplicateUploadItem(
                        filename=filename,
                        file_size=len(content),
                        message=f"Identical resume content already uploaded for this job ('{existing.original_filename}').",
                        existing_candidate_id=existing.id
                    ))
                    continue

                # Generate secure server-controlled filename
                stored_filename = f"{uuid.uuid4().hex}.{file_type}"
                stored_path = settings.UPLOAD_DIR / stored_filename

                # Write file to secure storage directory
                with open(stored_path, "wb") as f:
                    f.write(content)

                # Extract text using TextExtractionService
                extraction_result = TextExtractionService.extract_document_text(stored_path, file_type)

                status_val = "completed" if extraction_result.success else "failed"
                extracted_text = extraction_result.text if extraction_result.success else None
                error_msg = extraction_result.error_message if not extraction_result.success else None

                full_name = None
                email_addr = None
                phone_num = None
                location_str = None
                linkedin = None
                github = None
                portfolio = None
                struct_resume_json = None

                if extraction_result.success and extracted_text:
                    parsed = ResumeParser.parse_resume(extracted_text, filename=filename)
                    full_name = parsed.identity.full_name
                    email_addr = parsed.identity.email
                    phone_num = parsed.identity.phone
                    location_str = parsed.identity.location
                    linkedin = parsed.identity.linkedin_url
                    github = parsed.identity.github_url
                    portfolio = parsed.identity.portfolio_url
                    struct_resume_json = json.dumps(parsed.to_dict())

                # Persist candidate record
                candidate = Candidate(
                    job_id=job_id,
                    recruiter_id=recruiter_id,
                    original_filename=filename,
                    stored_filename=stored_filename,
                    file_type=file_type,
                    file_size=len(content),
                    file_hash=file_hash,
                    extracted_text=extracted_text,
                    processing_status=status_val,
                    error_message=error_msg,
                    full_name=full_name,
                    email=email_addr,
                    phone=phone_num,
                    location=location_str,
                    linkedin_url=linkedin,
                    github_url=github,
                    portfolio_url=portfolio,
                    structured_resume=struct_resume_json,
                    created_at=utc_now(),
                    updated_at=utc_now()
                )
                db.add(candidate)
                db.commit()
                db.refresh(candidate)

                candidate_resp = CandidateResponse(
                    id=candidate.id,
                    job_id=candidate.job_id,
                    recruiter_id=candidate.recruiter_id,
                    original_filename=candidate.original_filename,
                    file_type=candidate.file_type,
                    file_size=candidate.file_size,
                    processing_status=candidate.processing_status,
                    error_message=candidate.error_message,
                    full_name=candidate.full_name,
                    email=candidate.email,
                    phone=candidate.phone,
                    location=candidate.location,
                    linkedin_url=candidate.linkedin_url,
                    github_url=candidate.github_url,
                    portfolio_url=candidate.portfolio_url,
                    structured_resume=candidate.structured_resume,
                    created_at=candidate.created_at,
                    updated_at=candidate.updated_at,
                    job_title=job.title
                )
                uploaded_records.append(candidate_resp)

            except Exception as e:
                db.rollback()
                error_records.append(UploadErrorItem(
                    filename=filename,
                    reason=f"Processing failed: {str(e)}"
                ))

        return BatchUploadResponse(
            uploaded=uploaded_records,
            duplicates=duplicate_records,
            errors=error_records,
            total_uploaded=len(uploaded_records),
            total_duplicates=len(duplicate_records),
            total_errors=len(error_records)
        )

    @classmethod
    def get_candidates_for_job(
        cls,
        db: Session,
        job_id: int,
        recruiter_id: int,
        search: Optional[str] = None,
        status_filter: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> CandidateListResponse:
        """Retrieves paginated candidates for a specific recruiter's job with search and status filtering."""
        job = cls._verify_job_ownership(db, job_id, recruiter_id)

        query = db.query(Candidate).filter(
            Candidate.job_id == job_id,
            Candidate.recruiter_id == recruiter_id
        )

        if status_filter and status_filter.lower() != "all":
            query = query.filter(Candidate.processing_status == status_filter.strip().lower())

        if search and search.strip():
            term = f"%{search.strip()}%"
            query = query.filter(Candidate.original_filename.ilike(term))

        total = query.count()
        total_pages = max(1, (total + page_size - 1) // page_size)
        offset = (page - 1) * page_size

        candidates = query.order_by(
            Candidate.ranking_position.asc().nullslast(),
            Candidate.created_at.desc()
        ).offset(offset).limit(page_size).all()

        items = [
            CandidateResponse(
                id=c.id,
                job_id=c.job_id,
                recruiter_id=c.recruiter_id,
                original_filename=c.original_filename,
                file_type=c.file_type,
                file_size=c.file_size,
                extracted_text=c.extracted_text,
                processing_status=c.processing_status,
                error_message=c.error_message,
                relevance_score=c.relevance_score,
                fit_category=c.fit_category,
                ranking_position=c.ranking_position,
                predicted_probabilities=c.predicted_probabilities,
                model_version=c.model_version,
                created_at=c.created_at,
                updated_at=c.updated_at,
                job_title=job.title
            )
            for c in candidates
        ]

        return CandidateListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    @classmethod
    def get_all_candidates_for_recruiter(
        cls,
        db: Session,
        recruiter_id: int,
        job_id: Optional[int] = None,
        search: Optional[str] = None,
        status_filter: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> CandidateListResponse:
        """Retrieves all candidates for the authenticated recruiter, optionally filtered by job."""
        query = db.query(Candidate).filter(Candidate.recruiter_id == recruiter_id)

        if job_id is not None:
            query = query.filter(Candidate.job_id == job_id)

        if status_filter and status_filter.lower() != "all":
            query = query.filter(Candidate.processing_status == status_filter.strip().lower())

        if search and search.strip():
            term = f"%{search.strip()}%"
            query = query.filter(Candidate.original_filename.ilike(term))

        total = query.count()
        total_pages = max(1, (total + page_size - 1) // page_size)
        offset = (page - 1) * page_size

        candidates = query.order_by(Candidate.created_at.desc()).offset(offset).limit(page_size).all()

        items = []
        for c in candidates:
            items.append(CandidateResponse(
                id=c.id,
                job_id=c.job_id,
                recruiter_id=c.recruiter_id,
                original_filename=c.original_filename,
                file_type=c.file_type,
                file_size=c.file_size,
                extracted_text=c.extracted_text,
                processing_status=c.processing_status,
                error_message=c.error_message,
                relevance_score=c.relevance_score,
                fit_category=c.fit_category,
                ranking_position=c.ranking_position,
                predicted_probabilities=c.predicted_probabilities,
                model_version=c.model_version,
                created_at=c.created_at,
                updated_at=c.updated_at,
                job_title=c.job.title if c.job else None
            ))

        return CandidateListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    @classmethod
    def get_candidate_details(
        cls,
        db: Session,
        candidate_id: int,
        recruiter_id: int
    ) -> CandidateDetailResponse:
        """Retrieves candidate details including full extracted text with strict ownership check."""
        candidate = db.query(Candidate).filter(
            Candidate.id == candidate_id,
            Candidate.recruiter_id == recruiter_id
        ).first()

        if not candidate:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Candidate record not found or access denied."
            )

        return CandidateDetailResponse(
            id=candidate.id,
            job_id=candidate.job_id,
            recruiter_id=candidate.recruiter_id,
            original_filename=candidate.original_filename,
            file_type=candidate.file_type,
            file_size=candidate.file_size,
            processing_status=candidate.processing_status,
            error_message=candidate.error_message,
            relevance_score=candidate.relevance_score,
            fit_category=candidate.fit_category,
            ranking_position=candidate.ranking_position,
            predicted_probabilities=candidate.predicted_probabilities,
            model_version=candidate.model_version,
            created_at=candidate.created_at,
            updated_at=candidate.updated_at,
            job_title=candidate.job.title if candidate.job else None,
            extracted_text=candidate.extracted_text
        )

    @classmethod
    def delete_candidate(
        cls,
        db: Session,
        candidate_id: int,
        recruiter_id: int
    ) -> int:
        """Permanently deletes candidate record and safely purges the stored physical file."""
        candidate = db.query(Candidate).filter(
            Candidate.id == candidate_id,
            Candidate.recruiter_id == recruiter_id
        ).first()

        if not candidate:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Candidate record not found or access denied."
            )

        # Safely remove stored physical file
        stored_path = settings.UPLOAD_DIR / candidate.stored_filename
        if stored_path.exists() and stored_path.is_file():
            try:
                os.remove(stored_path)
            except OSError:
                pass  # Avoid transaction failure if file was already removed

        db.delete(candidate)
        db.commit()
        return candidate_id

    @classmethod
    def retry_extraction(
        cls,
        db: Session,
        candidate_id: int,
        recruiter_id: int
    ) -> CandidateDetailResponse:
        """Retries text extraction for an eligible candidate record."""
        candidate = db.query(Candidate).filter(
            Candidate.id == candidate_id,
            Candidate.recruiter_id == recruiter_id
        ).first()

        if not candidate:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Candidate record not found or access denied."
            )

        stored_path = settings.UPLOAD_DIR / candidate.stored_filename
        if not stored_path.exists():
            candidate.processing_status = "failed"
            candidate.error_message = "Stored resume document not found on disk. Please upload the file again."
            candidate.updated_at = utc_now()
            db.commit()
            db.refresh(candidate)
            return cls.get_candidate_details(db, candidate_id, recruiter_id)

        # Re-run extraction
        result = TextExtractionService.extract_document_text(stored_path, candidate.file_type)

        if result.success:
            candidate.extracted_text = result.text
            candidate.processing_status = "completed"
            candidate.error_message = None
            try:
                parsed = ResumeParser.parse_resume(result.text, filename=candidate.original_filename)
                candidate.full_name = parsed.identity.full_name
                candidate.email = parsed.identity.email
                candidate.phone = parsed.identity.phone
                candidate.location = parsed.identity.location
                candidate.linkedin_url = parsed.identity.linkedin_url
                candidate.github_url = parsed.identity.github_url
                candidate.portfolio_url = parsed.identity.portfolio_url
                candidate.structured_resume = json.dumps(parsed.to_dict())
            except Exception:
                pass
        else:
            candidate.extracted_text = None
            candidate.processing_status = "failed"
            candidate.error_message = result.error_message

        candidate.updated_at = utc_now()
        db.commit()
        db.refresh(candidate)

        return cls.get_candidate_details(db, candidate_id, recruiter_id)

    @classmethod
    def export_candidates_shortlist(
        cls,
        db: Session,
        job_id: int,
        recruiter_id: int,
        limit: int = 100,
        export_format: str = "xlsx"
    ) -> Response:
        job = cls._verify_job_ownership(db, job_id, recruiter_id)

        # Retrieve candidates in authoritative ranking order
        candidates = db.query(Candidate).filter(
            Candidate.job_id == job_id,
            Candidate.recruiter_id == recruiter_id,
            Candidate.processing_status == "completed"
        ).order_by(
            Candidate.ranking_position.asc().nullslast(),
            Candidate.relevance_score.desc().nullslast(),
            Candidate.id.asc()
        ).limit(limit).all()

        if not candidates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No completed candidates available to export for this job role."
            )

        slug = re.sub(r"[^A-Za-z0-9_-]", "_", job.title).strip("_") or "job"
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

        if export_format == "xlsx":
            file_bytes = ExportService.generate_excel_export(
                candidates=candidates,
                job_title=job.title,
                limit_requested=limit
            )
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"Verisift_Shortlist_{slug}_{date_str}.xlsx"
        elif export_format == "pdf":
            file_bytes = ExportService.generate_pdf_export(
                candidates=candidates,
                job_title=job.title,
                limit_requested=limit
            )
            media_type = "application/pdf"
            filename = f"Verisift_Shortlist_{slug}_{date_str}.pdf"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported export format '{export_format}'. Supported formats: 'xlsx', 'pdf'."
            )

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }

        return Response(content=file_bytes, media_type=media_type, headers=headers)
