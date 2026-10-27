from datetime import datetime, timedelta, timezone
from typing import Optional, Set
import jwt
import bcrypt
from app.core.config import settings

# In-memory revocation set for invalidated tokens (e.g., on logout)
# For scaled production, Redis or a database table can be swapped in.
_revoked_tokens: Set[str] = set()

def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt with a generated salt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a signed JWT access token with expiration."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": now,
        "iss": settings.APP_NAME
    })
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    """Decodes and validates a JWT access token, checking expiration and revocation."""
    if token in _revoked_tokens:
        return None
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            issuer=settings.APP_NAME
        )
        return payload
    except (jwt.PyJWTError, Exception):
        return None

def revoke_token(token: str) -> None:
    """Marks a token as revoked upon user logout."""
    _revoked_tokens.add(token)

def is_token_revoked(token: str) -> bool:
    """Checks if a token has been explicitly revoked."""
    return token in _revoked_tokens
