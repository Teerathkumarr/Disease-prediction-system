"""
MediSense — AI-Powered Early Disease Detection System
FastAPI Backend — Main Application
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from routers import cvd, diabetes, reports, auth
from database import engine, Base
from config import settings

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="MediSense API",
    description="AI-Powered Early Detection of Cardiovascular Disease and Diabetes",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router,     prefix="/api/auth",     tags=["Authentication"])
app.include_router(cvd.router,      prefix="/api/cvd",      tags=["Cardiovascular"])
app.include_router(diabetes.router, prefix="/api/diabetes", tags=["Diabetes"])
app.include_router(reports.router,  prefix="/api/reports",  tags=["Reports"])


@app.get("/")
async def root():
    return {
        "name": "MediSense API",
        "version": "1.0.0",
        "status": "online",
        "modules": ["cardiovascular", "diabetes"],
        "docs": "/api/docs"
    }

@app.get("/api/health")
async def health():
    return {"status": "healthy", "models_loaded": True}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
