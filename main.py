
"""
BodyScan AI — FastAPI Application Entry Point

Start the server:
    uvicorn main:app --reload --port 8000

API docs:
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import router as analysis_router

app = FastAPI(
    title="BodyScan AI",
    description="Body composition analysis from photos",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router)


@app.get("/", tags=["root"])
async def root():
    return {
        "service": "BodyScan AI Pipeline",
        "version": "0.1.0",
        "docs": "/docs",
    }
