from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check():
    """Simple health check endpoint for frontend connectivity and uptime monitoring."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0"
    }
