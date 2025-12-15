# server/app/models.py
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    plan: Mapped[str] = mapped_column(String(20), nullable=False, default="Free")
    stories_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


class Story(Base):
    __tablename__ = "stories"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    title = Column(String, nullable=False)  # пока можно оставить nullable
    config = Column(JSON, nullable=False)  # сюда кладём sys.json / payload

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    turns = relationship(
        "StoryTurn",
        back_populates="story",
        cascade="all, delete-orphan",
        order_by="StoryTurn.id",
    )


class StoryTurn(Base):
    __tablename__ = "story_turns"

    id = Column(Integer, primary_key=True, index=True)
    story_id = Column(Integer, ForeignKey("stories.id"), nullable=False, index=True)
    turns = Column(JSON, nullable=False, default=list)

    story = relationship("Story", back_populates="turns")