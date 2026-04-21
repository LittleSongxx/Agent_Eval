import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import engine, Base

# Import all models so they register with Base before create_all
import app.models  # noqa: F401


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup: create all tables
    Base.metadata.create_all(bind=engine)

    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    if settings.SEED_ON_STARTUP:
        # Run seed data (import here to avoid circular imports)
        from app.seed import run_seed
        from app.core.database import SessionLocal
        db = SessionLocal()
        try:
            run_seed(db)
        finally:
            db.close()

    yield

    # Shutdown: nothing to clean up for now


app = FastAPI(
    title="AI Evaluation Platform",
    description="Backend API for the AI Evaluation Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from app.api.llm_config import router as llm_config_router
from app.api.dataset import router as dataset_router
from app.api.metric import router as metric_router
from app.api.scenario import router as scenario_router
from app.api.evaluation import router as evaluation_router
from app.api.report import router as report_router

app.include_router(llm_config_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(metric_router, prefix="/api")
app.include_router(scenario_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")
app.include_router(report_router, prefix="/api")

# Static file serving for uploads
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
