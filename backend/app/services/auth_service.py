from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.user import User
from app.schemas.user import UserCreate
from app.schemas.auth import LoginRequest, AuthResponse
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    revoke_token
)

def normalize_email(email: str) -> str:
    """Normalizes an email address by stripping whitespace and converting to lowercase."""
    return email.strip().lower()

class AuthService:
    @staticmethod
    def register_user(db: Session, user_in: UserCreate) -> AuthResponse:
        normalized = normalize_email(str(user_in.email))
        
        # Check if confirmation password was provided and matches
        if user_in.confirm_password is not None and user_in.password != user_in.confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Passwords do not match."
            )
        
        # Check duplicate email
        existing = db.query(User).filter(User.email == normalized).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email address already exists."
            )
        
        # Hash password and store user
        hashed = hash_password(user_in.password)
        user = User(
            full_name=user_in.full_name.strip(),
            email=normalized,
            password_hash=hashed,
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Issue access token
        token = create_access_token({"sub": str(user.id), "email": user.email})
        return AuthResponse(access_token=token, user=user)

    @staticmethod
    def authenticate_user(db: Session, credentials: LoginRequest) -> AuthResponse:
        normalized = normalize_email(str(credentials.email))
        user = db.query(User).filter(User.email == normalized).first()
        
        # Generic error message to prevent email enumeration
        invalid_credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        if not user or not user.is_active:
            raise invalid_credentials_exception
        
        if not verify_password(credentials.password, user.password_hash):
            raise invalid_credentials_exception
        
        token = create_access_token({"sub": str(user.id), "email": user.email})
        return AuthResponse(access_token=token, user=user)

    @staticmethod
    def get_current_user_from_token(db: Session, token: str) -> User:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials or session expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        payload = decode_access_token(token)
        if not payload:
            raise credentials_exception
        
        user_id_str = payload.get("sub")
        if not user_id_str:
            raise credentials_exception
        
        try:
            user_id = int(user_id_str)
        except ValueError:
            raise credentials_exception
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise credentials_exception
        
        return user

    @staticmethod
    def logout_token(token: str) -> None:
        revoke_token(token)
