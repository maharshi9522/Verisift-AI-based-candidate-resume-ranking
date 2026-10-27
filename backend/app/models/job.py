from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.db.base import Base

def utc_now():
    return datetime.now(timezone.utc)

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    recruiter_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    department = Column(String(120), nullable=True)
    location = Column(String(120), nullable=True)
    employment_type = Column(String(50), nullable=False, default="Full-time")
    workplace_type = Column(String(50), nullable=False, default="On-site")
    description = Column(Text, nullable=False)
    required_skills = Column(JSON, nullable=False, default=list)
    preferred_skills = Column(JSON, nullable=False, default=list)
    minimum_experience = Column(Float, nullable=True)
    status = Column(String(50), nullable=False, default="draft")  # draft, active, archived
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    recruiter = relationship("User", backref="jobs")

    def __repr__(self) -> str:
        return f"<Job id={self.id} title='{self.title}' recruiter_id={self.recruiter_id} status='{self.status}'>"
