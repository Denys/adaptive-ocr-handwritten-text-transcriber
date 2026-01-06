from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.endpoints import ocr

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize things if needed
    yield
    # Shutdown: Cleanup

app = FastAPI(
    title="MultiScript OCR API",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(ocr.router, prefix="/api/ocr", tags=["ocr"])

@app.get("/health")
async def health_check():
    return {"status": "ok"}
