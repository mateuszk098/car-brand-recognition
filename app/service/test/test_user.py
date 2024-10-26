from unittest.mock import AsyncMock, MagicMock

import cv2 as cv
import numpy as np
import pytest
from fastapi import UploadFile
from pydantic import SecretStr
from sqlalchemy.orm import Session

from app import errors
from app.data.models import Task, User
from app.schema.user import Password, Role, UserCreate
from app.service import user as service
from app.test.utils import admin_user, db_clear_fixture, db_session_fixture, db_task_fixture, db_user_fixture


def get_user_create() -> UserCreate:
    return UserCreate(
        password=SecretStr("password"),
        confirm_password=SecretStr("password"),
        username="johndoe",
        email="johndoe@gmail.com",
        first_name="John",
        last_name="Doe",
        role=Role.user,
    )


def test_get_user_by_username_success(db_user_fixture: User, db_session_fixture: Session) -> None:
    result = service.get_user_by_username("johndoe", db_session_fixture)
    assert result.username == db_user_fixture.username
    assert result.email == db_user_fixture.email
    assert result.first_name == db_user_fixture.first_name
    assert result.last_name == db_user_fixture.last_name
    assert result.role == db_user_fixture.role


def test_get_user_by_username_fail(db_session_fixture: Session) -> None:
    with pytest.raises(errors.UserNotFoundError) as e:
        service.get_user_by_username("not_exists", db_session_fixture)
    assert e.value.detail == "User not_exists not found"


def test_get_users_success(db_session_fixture: Session) -> None:
    result = service.get_users(db_session_fixture)
    assert len(result) == 0


def test_create_user_success(db_session_fixture: Session, db_clear_fixture: None) -> None:
    user_create = get_user_create()
    service.create_user(user_create, db_session_fixture)
    user_response = service.get_user_by_username(user_create.username, db_session_fixture)
    assert user_create.username == user_response.username
    assert user_create.email == user_response.email
    assert user_create.first_name == user_response.first_name
    assert user_create.last_name == user_response.last_name
    assert user_create.role == user_response.role
    assert user_response.hashed_password is not None


def test_create_user_password_mismatch(db_session_fixture: Session) -> None:
    user_create = get_user_create()
    user_create.confirm_password = SecretStr("wrong_password")
    with pytest.raises(errors.PasswordMismatchError) as e:
        service.create_user(user_create, db_session_fixture)
    assert e.value.detail == "Passwords do not match"


def test_create_user_already_exists(db_user_fixture: User, db_session_fixture: Session) -> None:
    user_create = get_user_create()
    user_create.username = db_user_fixture.username
    user_create.email = "new_email@gmail.com"
    with pytest.raises(errors.UserAlreadyExistsError) as e:
        service.create_user(user_create, db_session_fixture)
    assert e.value.detail == f"User {user_create.username} already exists"


def test_create_email_already_exists(db_user_fixture: User, db_session_fixture: Session) -> None:
    user_create = get_user_create()
    user_create.username = "new_user"
    user_create.email = db_user_fixture.email
    with pytest.raises(errors.EmailAlreadyExistsError) as e:
        service.create_user(user_create, db_session_fixture)
    assert e.value.detail == f"Email {user_create.email} already exists"


def test_delete_user_by_username_success(db_user_fixture: User, db_session_fixture: Session) -> None:
    admin = admin_user()
    admin_password = Password(
        password=SecretStr("password"),
        confirm_password=SecretStr("password"),
    )
    service.delete_user_by_username(db_user_fixture.username, admin_password, admin, db_session_fixture)
    with pytest.raises(errors.UserNotFoundError) as e:
        service.get_user_by_username(db_user_fixture.username, db_session_fixture)
    assert e.value.detail == f"User {db_user_fixture.username} not found"


def test_delete_user_by_username_user_not_exists(db_session_fixture: Session) -> None:
    admin = admin_user()
    admin_password = Password(
        password=SecretStr("password"),
        confirm_password=SecretStr("password"),
    )
    with pytest.raises(errors.UserNotFoundError) as e:
        service.delete_user_by_username("not_exists", admin_password, admin, db_session_fixture)
    assert e.value.detail == "User not_exists not found"


def test_delete_user_by_username_admin_password_mismatch(db_user_fixture: User, db_session_fixture: Session) -> None:
    admin = admin_user()
    admin_password = Password(
        password=SecretStr("admin_password"),
        confirm_password=SecretStr("wrong_password"),
    )
    with pytest.raises(errors.PasswordMismatchError) as e:
        service.delete_user_by_username(db_user_fixture.username, admin_password, admin, db_session_fixture)
    assert e.value.detail == "Passwords do not match"


def test_delete_user_by_username_admin_invalid_password(db_user_fixture: User, db_session_fixture: Session) -> None:
    admin = admin_user()
    admin_password = Password(
        password=SecretStr("admin_password"),
        confirm_password=SecretStr("admin_password"),
    )
    with pytest.raises(errors.InvalidPasswordError) as e:
        service.delete_user_by_username(db_user_fixture.username, admin_password, admin, db_session_fixture)
    assert e.value.detail == "Invalid password"


def test_get_tasks_for_user_success(db_user_fixture: User, db_task_fixture: Task, db_session_fixture: Session) -> None:
    result = service.get_tasks_for_user(db_user_fixture.id, db_session_fixture)
    assert len(result) == 1
    assert result[0].name == db_task_fixture.name
    assert result[0].content == db_task_fixture.content
    assert result[0].brands == db_task_fixture.brands
    assert result[0].probs == db_task_fixture.probs
    assert result[0].time_created == db_task_fixture.time_created
    assert result[0].user_id == db_task_fixture.user_id
    assert result[0].id == db_task_fixture.id


def test_get_tasks_for_user_no_tasks(db_user_fixture: User, db_session_fixture: Session) -> None:
    result = service.get_tasks_for_user(db_user_fixture.id, db_session_fixture)
    assert len(result) == 0


@pytest.mark.asyncio()
async def test_create_task_success(db_user_fixture: User, db_session_fixture: Session) -> None:
    mock_upload = AsyncMock(spec=UploadFile)
    # Create a valid image byte content.
    _, image = cv.imencode(".jpg", np.zeros((720, 1280, 3), dtype=np.uint8))
    mock_upload.read.return_value = image.tobytes()
    mock_upload.filename = "test_image.jpg"
    mock_upload.content_type = "image/jpeg"
    mock_upload.file = MagicMock()

    mock_model = MagicMock()
    mock_model.return_value = [("brand1", 0.9), ("brand2", 0.1)]

    result = await service.create_task(mock_upload, 2, db_user_fixture.id, mock_model, db_session_fixture)

    assert result.name == mock_upload.filename
    assert result.content == mock_upload.content_type
    assert result.brands == "('brand1', 'brand2')"
    assert result.probs == "(0.9, 0.1)"
    assert result.user_id == db_user_fixture.id


@pytest.mark.asyncio()
async def test_create_task_image_decoding_error(db_user_fixture: User, db_session_fixture: Session) -> None:
    mock_upload = AsyncMock(spec=UploadFile)
    mock_upload.read.return_value = b"fake_image_data"
    mock_upload.filename = "test_image.jpg"
    mock_upload.content_type = "image/jpeg"
    mock_upload.file = MagicMock()

    mock_model = MagicMock()

    with pytest.raises(errors.ImageDecodingError) as e:
        await service.create_task(mock_upload, 2, db_user_fixture.id, mock_model, db_session_fixture)
    assert e.value.detail == "Cannot decode image"
