from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index, Float
from sqlalchemy.orm import relationship
from app.db.base import Base

def utc_now():
    return datetime.now(timezone.utc)

class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    recruiter_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True, index=True)
    file_type = Column(String(50), nullable=False)  # pdf, docx
    file_size = Column(Integer, nullable=False)      # bytes
    file_hash = Column(String(64), nullable=False, index=True)  # SHA-256
    extracted_text = Column(Text, nullable=True)
    processing_status = Column(String(50), nullable=False, default="processing", index=True)  # processing, completed, failed
    error_message = Column(Text, nullable=True)
    relevance_score = Column(Float, nullable=True)  # Ordinal score 0-100
    fit_category = Column(String(50), nullable=True)  # No Fit, Potential Fit, Good Fit
    ranking_position = Column(Integer, nullable=True)  # 1-based rank within job
    predicted_probabilities = Column(Text, nullable=True)  # JSON string of class probabilities
    model_version = Column(String(100), nullable=True)  # Model identifier used for scoring
    # Section-aware structured extraction & candidate identity
    full_name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    location = Column(String(255), nullable=True)
    linkedin_url = Column(String(500), nullable=True)
    github_url = Column(String(500), nullable=True)
    portfolio_url = Column(String(500), nullable=True)
    structured_resume = Column(Text, nullable=True)  # JSON representation of sections and chunks
    matched_skills = Column(Text, nullable=True)     # JSON list of matched skills / evidence strings
    explanation = Column(Text, nullable=True)        # JSON structured candidate assessment and evidence
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    job = relationship("Job", backref="candidates")
    recruiter = relationship("User", backref="candidates")

    __table_args__ = (
        Index("ix_candidates_job_recruiter", "job_id", "recruiter_id"),
        Index("ix_candidates_job_hash", "job_id", "file_hash"),
    )

    def __repr__(self) -> str:
        return f"<Candidate id={self.id} filename='{self.original_filename}' status='{self.processing_status}'>"
