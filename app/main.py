from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from time import perf_counter
from app.schemas import InferenceRequest, InferenceResponse, HealthResponse, StoryStepIn
from app.service import get_pipeline, Pipeline
from app.config import settings
from pydantic import BaseModel, EmailStr
from typing import Dict
import hashlib
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from .db import Base, engine, get_db
from . import models, story
from .storyteller_mini import generate_story_step
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


@app.post("/stories")
def create_story_endpoint(
    payload: dict,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    payload — это тот же JSON, который отправляет фронт:
    {
      "story_description": "...",
      "player_description": { "user": "..." },
      "NPC_description": [ ... ],
      "start_phrase": "..."
    }
    """
    title = payload.get("title")
    config = payload.get("config")
    db_story = story.create_story(
      db=db,
      owner_id=current_user.id,
      config=config,
      title=title,  # позже можно добавить поле названия на фронте
    )
    return {"id": db_story.id}


@app.get("/stories")
def list_stories_endpoint(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    stories = story.list_stories_for_user(db, owner_id=current_user.id)
    return [
        {
            "id": s.id,
            "title": s.title,
            "config": s.config,
            "created_at": s.created_at,
            "updated_at": s.updated_at,
        }
        for s in stories
    ]


@app.post("/stories/{story_id}/turns")
def add_turn_endpoint(
    story_id: int,
    body: dict,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # проверяем, что история принадлежит текущему пользователю
    db_story = story.get_story(db, story_id=story_id, owner_id=current_user.id)
    if not db_story:
        raise HTTPException(status_code=404, detail="История не найдена")

    user_text = body.get("user_text", "")
    model_text = body.get("model_text", "")
    if not user_text or not model_text:
        raise HTTPException(
            status_code=400,
            detail="user_text и model_text обязательны"
        )

    turn = story.add_turn(
        db=db,
        story_id=story_id,
        user_text=user_text,
        model_text=model_text,
    )

    return {"id": turn.id, "idx": turn.idx}


@app.get("/stories/{story_id}")
def get_story_endpoint(
    story_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Выдача полной истории по ID.
    Возвращает:
    - основные данные истории
    - список ходов
    """

    db_story = story.get_story(db, story_id=story_id, owner_id=current_user.id)
    if not db_story:
        raise HTTPException(status_code=404, detail="История не найдена")

    # 🔹 получаем ходы (если нужно)
    turns = story.get_turns(db, story_id=story_id)

    return {
        "id": db_story.id,
        "title": db_story.title,
        "config": db_story.config,
        "turns": [
            {
                "user": t.user,
                "assistant": t.assistant,
            }
            for t in turns
        ],
    }


@app.post("/api/story_step")
def story_step(payload: StoryStepIn, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    # payload должен содержать story_id и user_input
    text = generate_story_step(
        db=db,
        story_id=payload.story_id,
        user_id=current_user.id,
        user_input=payload.user_input,
    )
    return {"reply": text}