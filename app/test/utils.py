"""Test utilities for the application."""

from typing import Generator

import bcrypt
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.data.models import Base, Task, User
from app.schema.user import Role, UserSchema

APP_DB_URL = "sqlite:///:memory:"

engine = create_engine(
    APP_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # Single connection is sufficient for unit testing.
)
session_factory = sessionmaker(engine, autoflush=False, autocommit=False)
Base.metadata.create_all(engine)


def get_test_db() -> Generator[Session, None, None]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def get_test_user() -> User:
    return User(
        username="johndoe",
        email="johndoe@gmail.com",
        first_name="John",
        last_name="Doe",
        role=Role.admin,
        hashed_password=bcrypt.hashpw(b"password", bcrypt.gensalt()),
    )


def get_test_task() -> Task:
    return Task(
        user_id=1,
        name="test_task",
        content="test_content",
        brands="(BMW, Mercedes)",
        probs="(0.78, 0.11)",
    )


def admin_user() -> UserSchema:
    return UserSchema.model_validate(
        {
            "username": "johndoe",
            "email": "johndoe@gmail.com",
            "first_name": "John",
            "last_name": "Doe",
            "role": "admin",
            "time_created": "2021-01-01T00:00:00",
            "id": 1,
            "hashed_password": bcrypt.hashpw(b"password", bcrypt.gensalt()),
        }
    )


@pytest.fixture(scope="function")
def db_session_fixture() -> Generator[Session, None, None]:
    yield from get_test_db()


@pytest.fixture(scope="function")
def db_clear_fixture() -> Generator[None, None, None]:
    try:
        yield
    finally:
        with engine.connect() as connection:
            connection.execute(text("DELETE FROM users;"))
            connection.execute(text("DELETE FROM tasks;"))
            connection.commit()


@pytest.fixture(scope="function")
def db_user_fixture(db_session_fixture: Session) -> Generator[User, None, None]:
    user = get_test_user()
    db_session_fixture.add(user)
    db_session_fixture.commit()
    try:
        yield user
    finally:
        with engine.connect() as connection:
            connection.execute(text("DELETE FROM users;"))
            connection.commit()


@pytest.fixture(scope="function")
def db_task_fixture(db_session_fixture: Session) -> Generator[Task, None, None]:
    task = get_test_task()
    db_session_fixture.add(task)
    db_session_fixture.commit()
    try:
        yield task
    finally:
        with engine.connect() as connection:
            connection.execute(text("DELETE FROM tasks;"))
            connection.commit()
