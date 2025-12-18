# app/story_crud.py
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import func, or_

from . import models


# ---------- Истории ----------

def create_story(
    db: Session,
    owner_id: int | None,
    config: Dict[str, Any],
    title: Optional[str] = None,
    genre: str = "Fantasy",
) -> models.Story:
    """Создать историю и сохранить её конфиг (sys.json/payload)."""
    story = models.Story(
        owner_id=owner_id,
        title=title,
        genre=genre,
        config=config,
    )
    db.add(story)
    db.commit()
    db.refresh(story)
    return story


def get_story(db: Session, story_id: int, owner_id: int) -> Optional[models.Story]:
    """
    Получить историю по id, к которой есть доступ у пользователя:
    - либо его личная (owner_id = owner_id),
    - либо шаблонная (owner_id IS NULL).
    """
    return (
        db.query(models.Story)
        .filter(
            models.Story.id == story_id,
            or_(
                models.Story.owner_id == owner_id,
                models.Story.owner_id.is_(None),
            ),
        )
        .first()
    )


def list_stories_for_user(db: Session, owner_id: int) -> List[models.Story]:
    """Список всех историй пользователя (включая шаблоны)."""
    return (
        db.query(models.Story)
        .filter(
            or_(
                models.Story.owner_id == owner_id,
                models.Story.owner_id.is_(None),
            )
        )
        .order_by(models.Story.updated_at.desc())
        .all()
    )



# ---------- Ходы ----------

def add_turn(
    db: Session,
    story_id: int,
    user_text: str,
    model_text: str,
    yc_previous_response_id: str | None = None,
) -> models.StoryTurn:
    # Пытаемся найти существующую запись с ходами для этой истории и пользователя
    turn_row = (
        db.query(models.StoryTurn)
        .filter(
            models.StoryTurn.story_id == story_id,
        )
        .order_by(models.StoryTurn.id.asc())
        .first()
    )

    # Если ещё не было записей, создаём новую
    if not turn_row:
        turn_row = models.StoryTurn(
            story_id=story_id,
            turns=[],
            yc_previous_response_id=yc_previous_response_id,
        )
        db.add(turn_row)
        db.flush()  # чтобы получить id при необходимости

    # Обновляем массив ходов
    current_turns = list(turn_row.turns or [])
    current_turns.append(
        {
            "user_text": user_text,
            "model_text": model_text,
        }
    )
    turn_row.turns = current_turns

    # Сохраняем previous_response_id именно в story_turns (per user, per story)
    if yc_previous_response_id:
        turn_row.yc_previous_response_id = yc_previous_response_id

    # Обновим updated_at истории (без хранения yc_previous_response_id в stories)
    db.query(models.Story).filter(models.Story.id == story_id).update(
        {"updated_at": func.now()}
    )

    db.commit()
    db.refresh(turn_row)
    return turn_row


def get_turns(
    db: Session,
    story_id: int,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Получить последние N ходов истории (по умолчанию 50).

    Возвращает список словарей формата:
      {"user_text": "...", "model_text": "..."}
    """
    turn_row = (
        db.query(models.StoryTurn)
        .filter(
            models.StoryTurn.story_id == story_id,
        )
        .order_by(models.StoryTurn.id.asc())
        .first()
    )

    if not turn_row or not turn_row.turns:
        return []

    all_turns = list(turn_row.turns)
    if len(all_turns) <= limit:
        return all_turns
    # последние N ходов
    return all_turns[-limit:]