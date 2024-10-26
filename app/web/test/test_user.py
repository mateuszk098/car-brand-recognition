import tempfile
from unittest.mock import patch

from fastapi import status
from fastapi.testclient import TestClient

from app import errors
from app.data.models import Task, User
from app.main import app as fastapi_app
from app.test.utils import admin_user, db_session_fixture, db_task_fixture, db_user_fixture, get_test_db
from app.web.dependencies import get_current_user, get_db

fastapi_app.dependency_overrides[get_db] = get_test_db
fastapi_app.dependency_overrides[get_current_user] = admin_user

client = TestClient(fastapi_app)


def test_create_access_token_success(db_user_fixture: User) -> None:
    response = client.post(
        "/user/token",
        data={"username": db_user_fixture.username, "password": "password"},
    )
    assert response.status_code == status.HTTP_202_ACCEPTED
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"


def test_create_access_token_user_not_found(db_user_fixture: User) -> None:
    response = client.post(
        "/user/token",
        data={"username": "nonexistent_user", "password": "password"},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "User nonexistent_user not found"}


def test_create_access_token_invalid_password(db_user_fixture: User) -> None:
    response = client.post(
        "/user/token",
        data={"username": db_user_fixture.username, "password": "wrong_password"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": "Invalid password"}


def test_register_success() -> None:
    response = client.post(
        "/user/register",
        data={
            "username": "new_user",
            "password": "password",
            "confirm_password": "password",
            "email": "new_user@example.com",
            "first_name": "New",
            "last_name": "User",
            "role": "user",
        },
    )
    assert response.status_code == status.HTTP_201_CREATED
    user_data = response.json()
    assert user_data["username"] == "new_user"
    assert user_data["email"] == "new_user@example.com"
    assert user_data["first_name"] == "New"
    assert user_data["last_name"] == "User"


def test_register_password_mismatch() -> None:
    response = client.post(
        "/user/register",
        data={
            "username": "new_user",
            "password": "password",
            "confirm_password": "different_password",
            "email": "new_user@example.com",
            "first_name": "New",
            "last_name": "User",
            "role": "user",
        },
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json() == {"detail": "Passwords do not match"}


def test_register_user_already_exists(db_user_fixture: User) -> None:
    response = client.post(
        "/user/register",
        data={
            "username": db_user_fixture.username,
            "password": "password",
            "confirm_password": "password",
            "email": "new_email@example.com",
            "first_name": "New",
            "last_name": "User",
            "role": "user",
        },
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json() == {"detail": f"User {db_user_fixture.username} already exists"}


def test_register_email_already_exists(db_user_fixture: User) -> None:
    response = client.post(
        "/user/register",
        data={
            "username": "new_user",
            "password": "password",
            "confirm_password": "password",
            "email": db_user_fixture.email,
            "first_name": "New",
            "last_name": "User",
            "role": "user",
        },
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json() == {"detail": f"Email {db_user_fixture.email} already exists"}


def test_get_current_user_info(db_user_fixture: User) -> None:
    response = client.get("/user/info")
    assert response.status_code == status.HTTP_200_OK
    user_info = response.json()
    assert user_info["username"] == db_user_fixture.username
    assert user_info["email"] == db_user_fixture.email
    assert user_info["first_name"] == db_user_fixture.first_name
    assert user_info["last_name"] == db_user_fixture.last_name
    assert user_info["role"] == db_user_fixture.role
    assert user_info["id"] == db_user_fixture.id


def test_create_task_success(db_user_fixture: User, db_task_fixture: Task) -> None:
    with patch("app.service.user.create_task", return_value=db_task_fixture):
        with tempfile.TemporaryFile(suffix=".jpg") as fp:
            response = client.post(
                "/user/tasks/create",
                files={"image": fp},
                data={"topk": 5},  # type: ignore
            )
    assert response.status_code == status.HTTP_201_CREATED
    task_data = response.json()
    assert task_data["id"] == db_task_fixture.id
    assert task_data["user_id"] == db_user_fixture.id


def test_create_task_image_decoding_error() -> None:
    with patch("app.service.user.create_task", side_effect=errors.ImageDecodingError()):
        with tempfile.TemporaryFile(suffix=".pdf") as fp:
            response = client.post(
                "/user/tasks/create",
                files={"image": fp},
                data={"topk": 5},  # type: ignore
            )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json() == {"detail": "Cannot decode image"}


def test_get_user_tasks_success(db_user_fixture: User, db_task_fixture: Task) -> None:
    with patch("app.service.user.get_tasks_for_user", return_value=[db_task_fixture]):
        response = client.get("/user/tasks/")
    assert response.status_code == status.HTTP_200_OK
    tasks_data = response.json()
    assert len(tasks_data) == 1
    assert tasks_data[0]["id"] == db_task_fixture.id
    assert tasks_data[0]["user_id"] == db_user_fixture.id


def test_get_user_tasks_empty() -> None:
    with patch("app.service.user.get_tasks_for_user", return_value=[]):
        response = client.get("/user/tasks/")
    assert response.status_code == status.HTTP_200_OK
    tasks_data = response.json()
    assert tasks_data == []
