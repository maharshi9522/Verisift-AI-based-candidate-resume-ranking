from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.session import init_db
from app.api.routes import auth, health, jobs, candidates

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure SQLite tables exist
    init_db()
    yield
    # Shutdown logic if needed

app = FastAPI(
    title=settings.APP_NAME,
    description="Verisift - AI-powered resume-job matching and candidate intelligence backend API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers under /api prefix
app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(candidates.router, prefix="/api")

@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "tagline": "Find fit beyond keywords.",
        "description": "AI-powered resume-job matching and candidate intelligence.",
        "documentation": "/docs",
        "health": "/api/health"
    }
