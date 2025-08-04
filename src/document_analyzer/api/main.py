from fastapi import FastAPI
from ..core.config import settings
from ..core.logging import setup_logging
from .routes import analysis, batch

app = FastAPI(
    title="Document Analyzer API",
    description="API for analyzing documents with LLMs.",
    version="0.1.0",
)

app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(batch.router, prefix="/api/v1", tags=["batch"])

@app.on_event("startup")
def on_startup():
    setup_logging()

@app.get("/health")
def health_check():
    return {"status": "ok"}
