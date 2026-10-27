from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, ConfigDict

class UserBase(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=120, description="Recruiter full name")
    email: EmailStr = Field(..., description="Recruiter work email address")

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128, description="Password (min 8 characters)")
    confirm_password: str | None = Field(default=None, description="Password confirmation")

class UserResponse(BaseModel):
    id: int
    full_name: str
    email: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
