# app/story_crud.py
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import func

from . import models


# ---------- Истории ----------

def create_story(
    db: Session,
    owner_id: int,
    config: Dict[str, Any],
    title: Optional[str] = None,
) -> models.Story:
    """Создать историю и сохранить её конфиг (sys.json/payload)."""
    story = models.Story(
        owner_id=owner_id,
        title=title,
        config=config,
    )
    db.add(story)
    db.commit()
    db.refresh(story)
    return story


def get_story(db: Session, story_id: int, owner_id: int) -> Optional[models.Story]:
    """Получить историю по id, принадлежащую конкретному пользователю."""
    return (
        db.query(models.Story)
        .filter(
            models.Story.id == story_id,
            models.Story.owner_id == owner_id,
        )
        .first()
    )


def list_stories_for_user(db: Session, owner_id: int) -> List[models.Story]:
    """Список всех историй пользователя."""
    return (
        db.query(models.Story)
        .filter(models.Story.owner_id == owner_id)
        .order_by(models.Story.updated_at.desc())
        .all()
    )


# ---------- Ходы ----------

def add_turn(
    db: Session,
    story_id: int,
    user_text: str,
    model_text: str,
) -> models.StoryTurn:
    """
    Добавить ход к истории:
    - user_text: то, что написал пользователь
    - model_text: ответ нейросети
    """
    # определяем следующий индекс хода
    last_idx = (
        db.query(func.max(models.StoryTurn.idx))
        .filter(models.StoryTurn.story_id == story_id)
        .scalar()
    )
    next_idx = (last_idx or 0) + 1

    turn = models.StoryTurn(
        story_id=story_id,
        idx=next_idx,
        user_text=user_text,
        model_text=model_text,
    )
    db.add(turn)

    # обновим updated_at истории
    db.query(models.Story).filter(models.Story.id == story_id).update(
        {"updated_at": func.now()}
    )

    db.commit()
    db.refresh(turn)
    return turn


def get_turns(
    db: Session,
    story_id: int,
    limit: int = 50,
) -> List[models.StoryTurn]:
    """Получить последние N ходов истории (по умолчанию 50)."""
    return (
        db.query(models.StoryTurn)
        .filter(models.StoryTurn.story_id == story_id)
        .order_by(models.StoryTurn.idx.desc())
        .limit(limit)
        .all()[::-1]  # вернуть в прямом порядке
    )