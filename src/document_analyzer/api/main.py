from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.logging import setup_logging
from ..core.config import get_config
from ..storage.models import create_tables
from .routes import collections, jobs

app = FastAPI(
    title="Document Analyzer API",
    description="API for analyzing documents with LLMs.",
    version="0.1.0",
)

# Configure CORS with settings from config
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(collections.router, prefix="/api/v1", tags=["collections"])
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])


@app.on_event("startup")
def on_startup():
    setup_logging()
    create_tables()


@app.get("/health")
def health_check():
    return {"status": "ok"}
