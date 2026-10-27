from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

VALID_EMPLOYMENT_TYPES = {"Full-time", "Part-time", "Contract", "Internship", "Freelance"}
VALID_WORKPLACE_TYPES = {"On-site", "Hybrid", "Remote"}
VALID_STATUSES = {"draft", "active", "archived"}

def normalize_choice(value: str, valid_set: set[str], field_name: str) -> str:
    cleaned = value.strip()
    for valid in valid_set:
        if valid.lower() == cleaned.lower():
            return valid
    raise ValueError(f"Invalid {field_name}: '{value}'. Must be one of {sorted(list(valid_set))}")

def clean_skill_list(skills: Optional[List[str]]) -> List[str]:
    if not skills:
        return []
    cleaned: List[str] = []
    seen = set()
    for skill in skills:
        s = str(skill).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            cleaned.append(s)
    return cleaned

class JobBase(BaseModel):
    title: str = Field(..., min_length=2, max_length=255, description="Job role title")
    department: Optional[str] = Field(default=None, max_length=120, description="Department or division")
    location: Optional[str] = Field(default=None, max_length=120, description="Office location or geographic region")
    employment_type: str = Field(default="Full-time", description="Employment arrangement")
    workplace_type: str = Field(default="On-site", description="Workplace setup (On-site, Hybrid, Remote)")
    description: str = Field(..., min_length=10, description="Detailed job description and responsibilities")
    required_skills: List[str] = Field(default_factory=list, description="Must-have competencies")
    preferred_skills: List[str] = Field(default_factory=list, description="Nice-to-have competencies")
    minimum_experience: Optional[float] = Field(default=None, ge=0, le=50, description="Minimum years of experience")
    status: str = Field(default="draft", description="Role status (draft, active, archived)")

    @field_validator("title", mode="before")
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Title cannot be blank")
        return v.strip()

    @field_validator("description", mode="before")
    @classmethod
    def validate_description(cls, v: str) -> str:
        if not isinstance(v, str) or len(v.strip()) < 10:
            raise ValueError("Description must be at least 10 characters long")
        return v.strip()

    @field_validator("department", "location", mode="before")
    @classmethod
    def strip_optional_str(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("employment_type", mode="before")
    @classmethod
    def validate_employment_type(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Employment type must be a string")
        return normalize_choice(v, VALID_EMPLOYMENT_TYPES, "employment_type")

    @field_validator("workplace_type", mode="before")
    @classmethod
    def validate_workplace_type(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Workplace type must be a string")
        return normalize_choice(v, VALID_WORKPLACE_TYPES, "workplace_type")

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Status must be a string")
        cleaned = v.strip().lower()
        if cleaned not in VALID_STATUSES:
            raise ValueError(f"Invalid status: '{v}'. Must be one of {sorted(list(VALID_STATUSES))}")
        return cleaned

    @field_validator("required_skills", "preferred_skills", mode="before")
    @classmethod
    def validate_skills(cls, v: Optional[List[str]]) -> List[str]:
        return clean_skill_list(v)

class JobCreate(JobBase):
    pass

class JobUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=2, max_length=255)
    department: Optional[str] = Field(default=None, max_length=120)
    location: Optional[str] = Field(default=None, max_length=120)
    employment_type: Optional[str] = None
    workplace_type: Optional[str] = None
    description: Optional[str] = Field(default=None, min_length=10)
    required_skills: Optional[List[str]] = None
    preferred_skills: Optional[List[str]] = None
    minimum_experience: Optional[float] = Field(default=None, ge=0, le=50)
    status: Optional[str] = None

    @field_validator("title", mode="before")
    @classmethod
    def validate_title(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("Title cannot be blank")
            return v.strip()
        return v

    @field_validator("description", mode="before")
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            if not isinstance(v, str) or len(v.strip()) < 10:
                raise ValueError("Description must be at least 10 characters long")
            return v.strip()
        return v

    @field_validator("department", "location", mode="before")
    @classmethod
    def strip_optional_str(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("employment_type", mode="before")
    @classmethod
    def validate_employment_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return normalize_choice(v, VALID_EMPLOYMENT_TYPES, "employment_type")
        return v

    @field_validator("workplace_type", mode="before")
    @classmethod
    def validate_workplace_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return normalize_choice(v, VALID_WORKPLACE_TYPES, "workplace_type")
        return v

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            cleaned = str(v).strip().lower()
            if cleaned not in VALID_STATUSES:
                raise ValueError(f"Invalid status: '{v}'. Must be one of {sorted(list(VALID_STATUSES))}")
            return cleaned
        return v

    @field_validator("required_skills", "preferred_skills", mode="before")
    @classmethod
    def validate_skills(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            return clean_skill_list(v)
        return v

class JobResponse(BaseModel):
    id: int
    recruiter_id: int
    title: str
    department: Optional[str] = None
    location: Optional[str] = None
    employment_type: str
    workplace_type: str
    description: str
    required_skills: List[str]
    preferred_skills: List[str]
    minimum_experience: Optional[float] = None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class DeleteJobResponse(BaseModel):
    message: str = "Job role deleted successfully"
    id: int
