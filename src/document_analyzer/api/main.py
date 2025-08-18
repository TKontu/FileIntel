from fastapi import FastAPI

from ..core.logging import setup_logging
from .routes import collections

app = FastAPI(
    title="Document Analyzer API",
    description="API for analyzing documents with LLMs.",
    version="0.1.0",
)

app.include_router(collections.router, prefix="/api/v1", tags=["collections"])


@app.on_event("startup")
def on_startup():
    setup_logging()


@app.get("/health")
def health_check():
    return {"status": "ok"}
