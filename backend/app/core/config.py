import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

# Determine base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

class Settings(BaseSettings):
    APP_NAME: str = "Verisift"
    ENVIRONMENT: str = "development"
    SECRET_KEY: str = "verisift-dev-secret-key-32-chars-long-secure-token-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    DATABASE_URL: str = f"sqlite:///{DATA_DIR.as_posix()}/verisift.db"
    FRONTEND_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Resume storage and upload limits
    UPLOAD_DIR: Path = DATA_DIR / "resumes"
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB per file
    MAX_BATCH_SIZE: int = 20  # Max files per upload request
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure data and upload directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
