"""Test utilities for the application."""

from typing import Generator

import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.data.models import Base, Task, User
from app.schema.user import Role, UserCreate

APP_DB_URL = "sqlite:///:memory:"

engine = create_engine(
    APP_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # Single connection is sufficient for unit testing.
)
session_factory = sessionmaker(engine, autoflush=False, autocommit=False)
Base.metadata.create_all(engine)


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """Create a new database session for a test."""
    session = session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def db_user(db_session: Session) -> Generator[User, None, None]:
    """Create a new user for a test."""
    user = User(
        username="johndoe",
        email="johndoe@gmail.com",
        first_name="John",
        last_name="Doe",
        role="user",
        hashed_password=b"hashed_password",
    )
    db_session.add(user)
    db_session.commit()
    try:
        yield user
    finally:
        with engine.connect() as connection:
            connection.execute(text("DELETE FROM users;"))
            connection.commit()


@pytest.fixture(scope="function")
def user_create() -> Generator[UserCreate, None, None]:
    new_user = UserCreate(
        password=SecretStr("password"),
        confirm_password=SecretStr("password"),
        username="janedoe",
        email="janedoe@gmail.com",
        first_name="Jane",
        last_name="Doe",
        role=Role.user,
    )
    try:
        yield new_user
    finally:
        with engine.connect() as connection:
            connection.execute(text("DELETE FROM users;"))
            connection.commit()


@pytest.fixture(scope="function")
def db_task(db_session: Session) -> Generator[Task, None, None]:
    """Create a new task for a test."""
    task = Task(
        user_id=1,
        name="test_task",
        content="test_content",
        brands="BMW",
        probs="0.99",
    )
    db_session.add(task)
    db_session.commit()
    try:
        yield task
    finally:
        with engine.connect() as connection:
            connection.execute(text("DELETE FROM tasks;"))
            connection.commit()
