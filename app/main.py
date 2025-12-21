from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from time import perf_counter
from app.schemas import InferenceRequest, InferenceResponse, HealthResponse, StoryStepIn
from app.service import get_pipeline, Pipeline
from app.config import settings, MAX_FILE_SIZE
from app.storage import image_storage
from pydantic import BaseModel, EmailStr
from typing import Dict, Optional, Any, Literal
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
from .field_assistant import generate_field_value

class StoryUpdate(BaseModel):
    title: Optional[str] = None
    config: Optional[dict] = None

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
        plan="Free",
        stories_count=0,
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
    genre = payload.get("genre")
    db_story = story.create_story(
      db=db,
      owner_id=current_user.id,
      genre=genre,
      config=config,
      title=title,  # позже можно добавить поле названия на фронте
    )
    db.query(models.User).filter(models.User.id == current_user.id).update(
        {models.User.stories_count: models.User.stories_count + 1}
    )
    db.commit()
    return {"id": db_story.id}

@app.put("/stories/{story_id}")
def update_story_endpoint(
    story_id: int,
    payload: StoryUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Обновление существующей истории.
    Разрешено только для историй, принадлежащих текущему пользователю.
    Можно изменять title и config.
    """
    db_story = story.get_story(db, story_id=story_id, owner_id=current_user.id)
    if not db_story:
        # либо история не существует, либо не принадлежит пользователю
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Нет доступа к редактированию этой истории.",
        )

    # Обновляем только те поля, которые реально переданы
    if payload.title is not None:
        db_story.title = payload.title
    if payload.config is not None:
        db_story.config = payload.config

    db.commit()
    db.refresh(db_story)

    return {
        "id": db_story.id,
        "title": db_story.title,
        "config": db_story.config,
    }

@app.delete("/stories/{story_id}")
def delete_story_endpoint(
    story_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    db_story = (
        db.query(models.Story)
        .filter(models.Story.id == story_id)
        .first()
    )

    if not db_story:
        raise HTTPException(status_code=404, detail="История не найдена")

    # Нельзя удалять шаблоны
    if db_story.owner_id is None:
        raise HTTPException(status_code=403, detail="Нельзя удалять шаблонные истории")

    # Доступ только владельцу
    if db_story.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Нет доступа к удалению этой истории")

    # Удаляем связанные turns (на случай, если нет каскада)
    db.query(models.StoryTurn).filter(models.StoryTurn.story_id == story_id).delete()

    # Удаляем саму историю
    db.delete(db_story)

    # Обновляем счётчик историй пользователя (не ниже 0)
    if getattr(current_user, "stories_count", None) is not None:
        new_count = max(0, int(current_user.stories_count) - 1)
        db.query(models.User).filter(models.User.id == current_user.id).update(
            {models.User.stories_count: new_count}
        )

    db.commit()

    return {"ok": True}

@app.get("/stories")
def list_stories_endpoint(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    stories = story.list_stories_for_user(db, owner_id=current_user.id)
    return [
        {
            "id": s.id,
            "owner_id": s.owner_id,
            "title": s.title,
            "config": s.config,
            "created_at": s.created_at,
            "updated_at": s.updated_at,
            "genre": s.genre,
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

    return {
        "id": turn.id,
        "turns": turn.turns,
    }


@app.get("/stories/{story_id}")
def get_story_endpoint(
    story_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # Пытаемся получить историю (разрешаем owner_id == NULL — шаблон)
    db_story = (
        db.query(models.Story)
        .filter(models.Story.id == story_id)
        .first()
    )

    if not db_story:
        raise HTTPException(status_code=404, detail="История не найдена")

    # Если это шаблон — создаём (или переиспользуем) копию для пользователя
    if db_story.owner_id is None:
        # 2. Создаём копию шаблона
        copied_story = models.Story(
            owner_id=current_user.id,
            title=db_story.title,
            genre=db_story.genre,
            config=db_story.config,
        )
        db.add(copied_story)
        db.commit()
        db.refresh(copied_story)

        # инкремент счётчика историй пользователя
        db.query(models.User).filter(models.User.id == current_user.id).update(
            {models.User.stories_count: models.User.stories_count + 1}
        )
        db.commit()

        db_story = copied_story

        copy_id = copied_story.id
        # Проверка доступа: после копирования история обязана принадлежать пользователю
        if db_story.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Нет доступа к истории")
        return RedirectResponse(url=f"/stories/{copy_id}", status_code=307)
    
    # Получаем ходы уже КОПИИ истории
    turns = story.get_turns(db, story_id=db_story.id)

    return {
        "id": db_story.id,
        "title": db_story.title,
        "config": db_story.config,
        "turns": turns,
    }

@app.post("/stories/{story_id}/duplicate")
def duplicate_story_endpoint(
    story_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # Получаем исходную историю (шаблон или пользовательскую)
    source_story = (
        db.query(models.Story)
        .filter(models.Story.id == story_id)
        .first()
    )

    if not source_story:
        raise HTTPException(status_code=404, detail="История не найдена")

    # Если история пользовательская — проверяем доступ
    if source_story.owner_id is not None and source_story.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Нет доступа к истории")

    # Создаём копию
    copied_story = models.Story(
        owner_id=current_user.id,
        title=source_story.title,
        genre=source_story.genre,
        config=source_story.config,
    )

    # Если дублируем шаблон — сохраняем связь с ним
    if source_story.owner_id is None and hasattr(copied_story, "template_id"):
        copied_story.template_id = source_story.id

    db.add(copied_story)
    db.commit()
    db.refresh(copied_story)

    # Инкремент счётчика историй пользователя
    db.query(models.User).filter(models.User.id == current_user.id).update(
        {models.User.stories_count: models.User.stories_count + 1}
    )
    db.commit()

    return {"id": copied_story.id}

@app.post("/api/story_step")
def story_step(payload: StoryStepIn, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    # payload должен содержать story_id и user_input
    text = generate_story_step(
        db=db,
        story_id=payload.story_id,
        user_id=current_user.id,
        user_input=payload.user_input,
        mode=payload.mode,
    )
    return {"reply": text}

class FieldAssistantRequest(BaseModel):
    prompt: str
    genre: str

@app.post("/api/field_assistant")
async def field_assistant(req: FieldAssistantRequest, current_user=Depends(get_current_user)):
    """
    Генерация текста для одного из полей истории с помощью AI.
    """
    result = await generate_field_value(
        user_prompt=req.prompt,
        genre=req.genre,
    )
    return {"result": result}

class ProfileUserOut(BaseModel):
    id: int
    email: EmailStr
    username: str
    created_at: datetime
    plan: str
    stories_count: int

class ProfileResponse(BaseModel):
    user: ProfileUserOut

class UserUpdateRequest(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    password: Optional[str] = None

@app.get("/api/profile", response_model=ProfileResponse)
def get_profile(
    current_user=Depends(get_current_user),
):
    """
    Возвращает профиль текущего пользователя.
    """

    user_out = ProfileUserOut(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        created_at=current_user.created_at,
        plan=current_user.plan,
        stories_count=current_user.stories_count,
    )

    return ProfileResponse(user=user_out)

@app.post("/api/user/update", response_model=ProfileResponse)
def update_user(
    req: UserUpdateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Обновление данных текущего пользователя.
    Разрешённые поля: email, username, password.
    Поля опциональны — обновляются только переданные.
    """

    # email
    if req.email is not None:
        new_email = str(req.email).strip().lower()
        if new_email:
            existing = (
                db.query(models.User)
                .filter(models.User.email == new_email, models.User.id != current_user.id)
                .first()
            )
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User with this email already exists.",
                )
            current_user.email = new_email

    # username
    if req.username is not None:
        new_username = req.username.strip()
        if new_username:
            current_user.username = new_username

    # password
    if req.password is not None:
        new_password = req.password.strip()
        if new_password:
            if len(new_password) < 6:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Password must be at least 6 characters.",
                )
            current_user.password_hash = hash_password(new_password)

    db.add(current_user)
    db.commit()
    db.refresh(current_user)

    user_out = ProfileUserOut(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        created_at=current_user.created_at,
        plan=current_user.plan,
        stories_count=current_user.stories_count,
    )

    return ProfileResponse(user=user_out)


@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    if image.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(status_code=400, detail="Invalid image type")

    file_bytes = await image.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    url = image_storage.upload_image(
        file_bytes=file_bytes,
        content_type=image.content_type,
    )

    return {"url": url}


@app.get("/images/{image_name}")
def get_image(image_name: str):
    key = f"images/{image_name}"

    try:
        data, content_type = image_storage.get_image(key)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")

    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=86400",
        },
    )