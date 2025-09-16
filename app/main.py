"""FastAPI application for the GUIDE system.

Exposes REST APIs for entity graph construction/optimization and question
answering; additional administrative endpoints provide health checks and
configuration insights.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from pathlib import Path

from app.api.v1.api import api_router
from app.core.config import settings

# Create FastAPI application
app = FastAPI(
    title="Research_CODE API",
    description="""
    A comprehensive entity graph construction and query processing system.

    ## Capabilities

    * **Graph Processing**: Build, process, and optimize entity graphs
    * **Query Processing**: Question answering with agentic workflows
    * **Administration**: Health metrics and configuration inspection

    ## Data Flow

    Entity extraction → Graph building → Processing → Optimization → Query processing
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint providing basic system information."""
    return {
        "message": "Research_CODE API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "research-code-api",
        "version": "1.0.0"
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
