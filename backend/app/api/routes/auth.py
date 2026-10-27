from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.user import UserCreate, UserResponse
from app.schemas.auth import LoginRequest, AuthResponse, LogoutResponse
from app.services.auth_service import AuthService
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthService.get_current_user_from_token(db, credentials.credentials)

@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(user_in: UserCreate, db: Session = Depends(get_db)):
    """Registers a new recruiter account and issues an authentication session."""
    return AuthService.register_user(db, user_in)

@router.post("/login", response_model=AuthResponse)
def login(credentials: LoginRequest, db: Session = Depends(get_db)):
    """Authenticates a recruiter using email and password."""
    return AuthService.authenticate_user(db, credentials)

@router.get("/me", response_model=UserResponse)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Retrieves the authenticated recruiter's profile data."""
    return current_user

@router.post("/logout", response_model=LogoutResponse)
def logout(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    """Revokes the current authentication token."""
    if credentials and credentials.credentials:
        AuthService.logout_token(credentials.credentials)
    return LogoutResponse(message="Logged out successfully", success=True)
