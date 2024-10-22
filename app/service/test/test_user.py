import pytest
from pydantic import SecretStr
from sqlalchemy.orm import Session

from app import errors
from app.data.models import Task, User
from app.schema.user import Role, UserCreate
from app.service import user as service
from app.test.utils import db_session, db_task, db_user, user_create


def test_get_user_by_username_success(db_user: User, db_session: Session) -> None:
    result = service.get_user_by_username("johndoe", db_session)
    assert result.username == db_user.username
    assert result.email == db_user.email
    assert result.first_name == db_user.first_name
    assert result.last_name == db_user.last_name
    assert result.role == db_user.role


def test_get_user_by_username_fail(db_session: Session) -> None:
    with pytest.raises(errors.UserNotFoundError) as e:
        service.get_user_by_username("not_exists", db_session)
    assert e.value.detail == "User not_exists not found"


def test_get_users_success(db_user: User, db_session: Session) -> None:
    result = service.get_users(db_session)
    assert len(result) == 1


def test_create_user_success(user_create: UserCreate, db_session: Session) -> None:
    service.create_user(user_create, db_session)
    user_response = service.get_user_by_username(user_create.username, db_session)
    assert user_create.username == user_response.username
    assert user_create.email == user_response.email
    assert user_create.first_name == user_response.first_name
    assert user_create.last_name == user_response.last_name
    assert user_create.role == user_response.role
    assert user_response.hashed_password is not None


def test_create_user_password_mismatch(user_create: UserCreate, db_session: Session) -> None:
    user_create.confirm_password = SecretStr("wrong_password")
    with pytest.raises(errors.PasswordMismatchError) as e:
        service.create_user(user_create, db_session)
    assert e.value.detail == "Passwords do not match"


def test_create_user_already_exists(user_create: UserCreate, db_user: User, db_session: Session) -> None:
    user_create.username = db_user.username
    with pytest.raises(errors.UserAlreadyExistsError) as e:
        service.create_user(user_create, db_session)
    assert e.value.detail == f"User {user_create.username} already exists"


def test_create_email_already_exists(user_create: UserCreate, db_user: User, db_session: Session) -> None:
    user_create.email = db_user.email
    with pytest.raises(errors.EmailAlreadyExistsError) as e:
        service.create_user(user_create, db_session)
    assert e.value.detail == f"Email {user_create.email} already exists"
