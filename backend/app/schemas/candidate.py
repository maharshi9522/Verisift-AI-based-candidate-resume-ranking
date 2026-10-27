from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

class CandidateBase(BaseModel):
    original_filename: str
    file_type: str
    file_size: int
    processing_status: str
    error_message: Optional[str] = None

class CandidateResponse(CandidateBase):
    id: int
    job_id: int
    recruiter_id: int
    extracted_text: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    job_title: Optional[str] = None
    relevance_score: Optional[float] = None
    fit_category: Optional[str] = None
    ranking_position: Optional[int] = None
    predicted_probabilities: Optional[str] = None
    model_version: Optional[str] = None
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    matched_skills: Optional[str] = None
    structured_resume: Optional[str] = None
    explanation: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class CandidateDetailResponse(CandidateResponse):
    pass

class RankJobResponse(BaseModel):
    job_id: int
    job_title: str
    model_used: str
    scoring_note: str = "Ordinal continuous relevance score (0-100) based on fit-class probability distribution."
    total_candidates_ranked: int
    ranked_candidates: List[CandidateResponse]
    ranked_at: datetime

class CandidateListResponse(BaseModel):
    items: List[CandidateResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

class DuplicateUploadItem(BaseModel):
    filename: str
    file_size: int
    message: str
    existing_candidate_id: int

class UploadErrorItem(BaseModel):
    filename: str
    reason: str

class BatchUploadResponse(BaseModel):
    uploaded: List[CandidateResponse]
    duplicates: List[DuplicateUploadItem]
    errors: List[UploadErrorItem]
    total_uploaded: int
    total_duplicates: int
    total_errors: int

class DeleteCandidateResponse(BaseModel):
    message: str = "Candidate record deleted successfully"
    id: int
