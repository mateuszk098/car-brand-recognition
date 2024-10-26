import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.data.models import Task, User
from app.data.repositories import TaskRepository, UserRepository
from app.errors import RecordNotFoundError
from app.test.utils import (
    db_clear_fixture,
    db_session_fixture,
    db_task_fixture,
    db_user_fixture,
    get_test_task,
    get_test_user,
)


def test_get_user_by_username_success(db_user_fixture: User, db_session_fixture: Session) -> None:
    result = UserRepository.get_user_by_username("johndoe", db_session_fixture)
    assert result.username == db_user_fixture.username
    assert result.email == db_user_fixture.email
    assert result.first_name == db_user_fixture.first_name
    assert result.last_name == db_user_fixture.last_name
    assert result.role == db_user_fixture.role


def test_get_user_by_username_fail(db_session_fixture: Session) -> None:
    with pytest.raises(RecordNotFoundError) as e:
        UserRepository.get_user_by_username("not_exists", db_session_fixture)
    assert e.value.detail == "User not found"


def test_get_user_by_email_success(db_user_fixture: User, db_session_fixture: Session) -> None:
    result = UserRepository.get_user_by_email("johndoe@gmail.com", db_session_fixture)
    assert result.username == db_user_fixture.username
    assert result.email == db_user_fixture.email
    assert result.first_name == db_user_fixture.first_name
    assert result.last_name == db_user_fixture.last_name
    assert result.role == db_user_fixture.role


def test_get_user_by_email_fail(db_session_fixture: Session) -> None:
    with pytest.raises(RecordNotFoundError) as e:
        UserRepository.get_user_by_email("notexist@gmail.com", db_session_fixture)
    assert e.value.detail == "User not found"


def test_get_user_by_id_success(db_user_fixture: User, db_session_fixture: Session) -> None:
    result = UserRepository.get_user_by_id(1, db_session_fixture)
    assert result.username == db_user_fixture.username
    assert result.email == db_user_fixture.email
    assert result.first_name == db_user_fixture.first_name
    assert result.last_name == db_user_fixture.last_name
    assert result.role == db_user_fixture.role


def test_get_user_by_id_fail(db_session_fixture: Session) -> None:
    with pytest.raises(RecordNotFoundError) as e:
        UserRepository.get_user_by_id(999, db_session_fixture)
    assert e.value.detail == "User not found"


def test_get_users_success(db_session_fixture: Session) -> None:
    result = UserRepository.get_users(db_session_fixture)
    assert len(result) == 0


def test_create_user_success(db_session_fixture: Session, db_clear_fixture: None) -> None:
    user = get_test_user()
    UserRepository.create_user(user, db_session_fixture)
    db_user_fixture = UserRepository.get_user_by_username(user.username, db_session_fixture)
    assert user.username == db_user_fixture.username
    assert user.email == db_user_fixture.email
    assert user.first_name == db_user_fixture.first_name
    assert user.last_name == db_user_fixture.last_name
    assert user.role == db_user_fixture.role


def test_create_user_fail_username_exists(db_user_fixture: User, db_session_fixture: Session) -> None:
    user = get_test_user()
    user.email = "new_email@gmail.com"  # Change email to avoid email conflict.
    with pytest.raises(IntegrityError) as e:
        UserRepository.create_user(user, db_session_fixture)
    assert "UNIQUE constraint failed: users.username" in str(e.value)


def test_create_user_fail_email_exists(db_user_fixture: User, db_session_fixture: Session) -> None:
    user = get_test_user()
    user.username = "new_user"  # Change username to avoid username conflict.
    with pytest.raises(IntegrityError) as e:
        UserRepository.create_user(user, db_session_fixture)
    assert "UNIQUE constraint failed: users.email" in str(e.value)


def test_delete_user_success(db_user_fixture: User, db_session_fixture: Session) -> None:
    UserRepository.delete_user(db_user_fixture, db_session_fixture)
    with pytest.raises(RecordNotFoundError) as e:
        UserRepository.get_user_by_id(db_user_fixture.id, db_session_fixture)
    assert e.value.detail == "User not found"


def test_get_task_by_id_success(db_task_fixture: Task, db_session_fixture: Session) -> None:
    task = TaskRepository.get_task_by_id(1, db_session_fixture)
    assert task.name == db_task_fixture.name
    assert task.content == db_task_fixture.content
    assert task.brands == db_task_fixture.brands
    assert task.probs == db_task_fixture.probs


def test_get_task_by_id_fail(db_session_fixture: Session) -> None:
    with pytest.raises(RecordNotFoundError) as e:
        TaskRepository.get_task_by_id(999, db_session_fixture)
    assert e.value.detail == "Task not found"


def test_get_tasks_for_user_success(db_user_fixture: User, db_task_fixture: Task, db_session_fixture: Session) -> None:
    tasks = TaskRepository.get_tasks_for_user(1, db_session_fixture)
    assert len(tasks) == 1
    assert tasks[0].user_id == db_user_fixture.id
    assert tasks[0].name == db_task_fixture.name
    assert tasks[0].content == db_task_fixture.content
    assert tasks[0].brands == db_task_fixture.brands
    assert tasks[0].probs == db_task_fixture.probs


def test_get_tasks_for_user_fail(db_session_fixture: Session) -> None:
    tasks = TaskRepository.get_tasks_for_user(999, db_session_fixture)
    assert len(tasks) == 0


def test_create_task_success(db_session_fixture: Session, db_clear_fixture: None) -> None:
    task = get_test_task()
    TaskRepository.create_task(task, db_session_fixture)
    db_task = TaskRepository.get_task_by_id(task.id, db_session_fixture)
    assert task.user_id == db_task.user_id
    assert task.name == db_task.name
    assert task.content == db_task.content
    assert task.brands == db_task.brands
    assert task.probs == db_task.probs
