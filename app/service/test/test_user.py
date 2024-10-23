import os
from unittest.mock import AsyncMock, MagicMock

import cv2 as cv
import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv
from fastapi import UploadFile
from pydantic import SecretStr
from sqlalchemy.orm import Session

from app import errors
from app.data.models import Task, User
from app.schema.user import Password, UserCreate, UserSchema
from app.service import user as service
from app.test.utils import admin_user, db_session, db_task, db_user, user_create

load_dotenv(find_dotenv())

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


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


def test_delete_user_by_username_success(admin_user: UserSchema, db_user: User, db_session: Session) -> None:
    admin_password = Password(
        password=SecretStr("password"),
        confirm_password=SecretStr("password"),
    )
    service.delete_user_by_username(db_user.username, admin_password, admin_user, db_session)
    with pytest.raises(errors.UserNotFoundError) as e:
        service.get_user_by_username(db_user.username, db_session)
    assert e.value.detail == f"User {db_user.username} not found"


def test_delete_user_by_username_user_not_exists(admin_user: UserSchema, db_session: Session) -> None:
    admin_password = Password(
        password=SecretStr("password"),
        confirm_password=SecretStr("password"),
    )
    with pytest.raises(errors.UserNotFoundError) as e:
        service.delete_user_by_username("not_exists", admin_password, admin_user, db_session)
    assert e.value.detail == "User not_exists not found"


def test_delete_user_by_username_admin_password_mismatch(
    admin_user: UserSchema, db_user: User, db_session: Session
) -> None:
    admin_password = Password(
        password=SecretStr("admin_password"),
        confirm_password=SecretStr("wrong_password"),
    )
    with pytest.raises(errors.PasswordMismatchError) as e:
        service.delete_user_by_username(db_user.username, admin_password, admin_user, db_session)
    assert e.value.detail == "Passwords do not match"


def test_delete_user_by_username_admin_invalid_password(
    admin_user: UserSchema, db_user: User, db_session: Session
) -> None:
    admin_password = Password(
        password=SecretStr("admin_password"),
        confirm_password=SecretStr("admin_password"),
    )
    with pytest.raises(errors.InvalidPasswordError) as e:
        service.delete_user_by_username(db_user.username, admin_password, admin_user, db_session)
    assert e.value.detail == "Invalid password"


def test_get_tasks_for_user_success(db_user: User, db_task: Task, db_session: Session) -> None:
    result = service.get_tasks_for_user(db_user.id, db_session)
    assert len(result) == 1
    assert result[0].name == db_task.name
    assert result[0].content == db_task.content
    assert result[0].brands == db_task.brands
    assert result[0].probs == db_task.probs
    assert result[0].time_created == db_task.time_created
    assert result[0].user_id == db_task.user_id
    assert result[0].id == db_task.id


def test_get_tasks_for_user_no_tasks(db_user: User, db_session: Session) -> None:
    result = service.get_tasks_for_user(db_user.id, db_session)
    assert len(result) == 0


@pytest.mark.asyncio()
async def test_create_task_success(db_user: User, db_session: Session) -> None:
    mock_upload = AsyncMock(spec=UploadFile)
    # Create a valid image byte content.
    _, image = cv.imencode(".jpg", np.zeros((720, 1280, 3), dtype=np.uint8))
    mock_upload.read.return_value = image.tobytes()
    mock_upload.filename = "test_image.jpg"
    mock_upload.content_type = "image/jpeg"
    mock_upload.file = MagicMock()

    mock_model = MagicMock()
    mock_model.return_value = [("brand1", 0.9), ("brand2", 0.1)]

    result = await service.create_task(mock_upload, 2, db_user.id, mock_model, db_session)

    assert result.name == mock_upload.filename
    assert result.content == mock_upload.content_type
    assert result.brands == "('brand1', 'brand2')"
    assert result.probs == "(0.9, 0.1)"
    assert result.user_id == db_user.id


@pytest.mark.asyncio()
async def test_create_task_image_decoding_error(db_user: User, db_session: Session) -> None:
    mock_upload = AsyncMock(spec=UploadFile)
    mock_upload.read.return_value = b"fake_image_data"
    mock_upload.filename = "test_image.jpg"
    mock_upload.content_type = "image/jpeg"
    mock_upload.file = MagicMock()

    mock_model = MagicMock()

    with pytest.raises(errors.ImageDecodingError) as e:
        await service.create_task(mock_upload, 2, db_user.id, mock_model, db_session)
    assert e.value.detail == "Cannot decode image"
