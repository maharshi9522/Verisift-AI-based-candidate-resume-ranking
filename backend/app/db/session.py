from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from app.core.config import settings
from app.db.base import Base

# SQLite configuration
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """Dependency that provides a transactional database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    """Initializes tables in the database and applies lightweight column migrations for SQLite."""
    # Import all models here so that Base knows about them before create_all
    import app.models.user       # noqa: F401
    import app.models.job        # noqa: F401
    import app.models.candidate  # noqa: F401
    from sqlalchemy import text

    Base.metadata.create_all(bind=engine)

    # Automatic migration for candidates table columns in SQLite
    if "sqlite" in settings.DATABASE_URL:
        with engine.connect() as conn:
            try:
                result = conn.execute(text("PRAGMA table_info(candidates)")).fetchall()
                existing_cols = {row[1] for row in result}
                new_cols = [
                    ("relevance_score", "REAL"),
                    ("fit_category", "VARCHAR(50)"),
                    ("ranking_position", "INTEGER"),
                    ("predicted_probabilities", "TEXT"),
                    ("model_version", "VARCHAR(100)"),
                    ("full_name", "VARCHAR(255)"),
                    ("email", "VARCHAR(255)"),
                    ("phone", "VARCHAR(50)"),
                    ("location", "VARCHAR(255)"),
                    ("linkedin_url", "VARCHAR(500)"),
                    ("github_url", "VARCHAR(500)"),
                    ("portfolio_url", "VARCHAR(500)"),
                    ("structured_resume", "TEXT"),
                    ("matched_skills", "TEXT"),
                    ("explanation", "TEXT")
                ]
                for col_name, col_type in new_cols:
                    if col_name not in existing_cols:
                        conn.execute(text(f"ALTER TABLE candidates ADD COLUMN {col_name} {col_type}"))
                conn.commit()
            except Exception:
                pass
