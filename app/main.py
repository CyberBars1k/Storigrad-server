from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from time import perf_counter
from app.schemas import InferenceRequest, InferenceResponse, HealthResponse
from app.service import get_pipeline, Pipeline
from app.config import settings
from pydantic import BaseModel, EmailStr
from typing import Dict
import hashlib
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from .db import Base, engine, get_db
from . import models
from jose import jwt
from datetime import timedelta
from .auth import (
    RegisterRequest,
    LoginRequest,
    AuthResponse,
    hash_password,
    create_jwt,
    get_current_user,
)

app = FastAPI(title="Storigrad API", version="0.1.0")

Base.metadata.create_all(bind=engine)

app.add_middleware(
CORSMiddleware,
allow_origins=settings.CORS_ALLOW_ORIGINS,
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)) -> AuthResponse:
    email = req.email.lower()

    existing = db.query(models.User).filter(models.User.email == email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists.",
        )

    user = models.User(
        email=email,
        username=req.username,
        password_hash=hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return AuthResponse(message="User registered successfully.", token="")


@app.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    email = req.email.lower()
    user = db.query(models.User).filter(models.User.email == email).first()

    if not user or user.password_hash != hash_password(req.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token = create_jwt(user.id)
    return AuthResponse(message="Login successful.", token=token)
