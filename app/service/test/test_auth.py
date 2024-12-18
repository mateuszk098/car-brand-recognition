import os
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import jwt
import pytest
from dotenv import find_dotenv, load_dotenv
from fastapi import HTTPException, status
from jwt.exceptions import ExpiredSignatureError
from sqlalchemy.orm import Session

from app import errors
from app.service import auth as service
from app.test.utils import admin_user, db_session_fixture

load_dotenv(find_dotenv())

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


def test_authenticate_user_success(db_session_fixture: Session) -> None:
    admin = admin_user()
    password = "password"
    with patch("app.service.auth.get_user_by_username", return_value=admin):
        with patch("app.service.auth.verify_password", return_value=True):
            result = service.authenticate_user(admin.username, password, db_session_fixture)
    assert result == admin


def test_authenticate_user_invalid_password(db_session_fixture: Session) -> None:
    admin = admin_user()
    password = "wrongpassword"
    with patch("app.service.auth.get_user_by_username", return_value=admin):
        with patch("app.service.auth.verify_password", return_value=False):
            with pytest.raises(errors.InvalidPasswordError) as e:
                service.authenticate_user(admin.username, password, db_session_fixture)
    assert e.value.detail == "Invalid password"


def test_create_access_token_success() -> None:
    admin = admin_user()
    expires = timedelta(minutes=15)
    token = service.create_access_token(admin.username, expires)
    decoded_token = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert decoded_token["sub"] == admin.username
    assert "exp" in decoded_token


def test_create_access_token_expired() -> None:
    admin = admin_user()
    expires = timedelta(seconds=1)
    token = service.create_access_token(admin.username, expires)
    decoded_token = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert decoded_token["sub"] == admin.username
    assert "exp" in decoded_token
    # Wait for the token to expire
    time.sleep(2)
    with pytest.raises(ExpiredSignatureError) as e:
        jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert str(e.value) == "Signature has expired"


def test_extract_username_from_token_success() -> None:
    admin = admin_user()
    expires = timedelta(minutes=15)
    token = service.create_access_token(admin.username, expires)
    username = service.extract_username_from_token(token)
    assert username == admin.username


def test_extract_username_from_token_expired() -> None:
    admin = admin_user()
    expires = timedelta(seconds=1)
    token = service.create_access_token(admin.username, expires)
    # Wait for the token to expire
    time.sleep(2)
    with pytest.raises(errors.InvalidCredentialsError) as e:
        service.extract_username_from_token(token)
    assert e.value.detail == "Invalid authentication credentials"


def test_extract_username_from_token_invalid_token() -> None:
    invalid_token = "invalid.token.here"
    with pytest.raises(errors.InvalidCredentialsError) as e:
        service.extract_username_from_token(invalid_token)
    assert e.value.detail == "Invalid authentication credentials"


def test_extract_username_from_token_no_username() -> None:
    expires = timedelta(minutes=15)
    payload = {"exp": datetime.now(UTC) + expires}
    token = jwt.encode(payload, JWT_SECRET_KEY, JWT_ALGORITHM)
    with pytest.raises(errors.InvalidCredentialsError) as e:
        service.extract_username_from_token(token)
    assert e.value.detail == "Invalid authentication credentials"


@pytest.mark.asyncio()
async def test_get_current_user_success(db_session_fixture: Session) -> None:
    admin = admin_user()
    token = service.create_access_token(admin.username, timedelta(minutes=15))
    with patch("app.service.auth.extract_username_from_token", return_value=admin.username):
        with patch("app.service.auth.get_user_by_username", return_value=admin):
            result = await service.get_current_user(token, db_session_fixture)
    assert result == admin


@pytest.mark.asyncio()
async def test_get_current_user_invalid_token(db_session_fixture: Session) -> None:
    invalid_token = "invalid.token.here"
    with patch("app.service.auth.extract_username_from_token", side_effect=errors.InvalidCredentialsError):
        with pytest.raises(HTTPException) as e:
            await service.get_current_user(invalid_token, db_session_fixture)
    assert e.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert e.value.detail == "Invalid authentication credentials"
