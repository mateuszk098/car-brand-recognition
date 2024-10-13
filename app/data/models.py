"""Database models for the application."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class User(Base):
    """Database model of a user."""

    __tablename__ = "users"

    time_created: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True)
    first_name: Mapped[str] = mapped_column(String)
    last_name: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)
    hashed_password: Mapped[bytes] = mapped_column(LargeBinary)


class Task(Base):
    """Database model of a deep learning model's prediction."""

    __tablename__ = "tasks"

    time_created: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(String)
    brands: Mapped[str] = mapped_column(String)
    probs: Mapped[str] = mapped_column(String)
