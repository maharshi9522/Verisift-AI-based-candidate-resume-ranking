from pydantic import BaseModel, EmailStr, Field
from app.schemas.user import UserResponse

class LoginRequest(BaseModel):
    email: EmailStr = Field(..., description="Recruiter work email")
    password: str = Field(..., min_length=1, description="Password")

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class LogoutResponse(BaseModel):
    message: str = "Logged out successfully"
    success: bool = True
