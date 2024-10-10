import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.errors import RecordNotFoundError
from app.test.utils import db_session, db_task, db_user

from ..models import Task, User
from ..repositories import TaskRepository, UserRepository


def test_get_user_by_username_success(db_user: User, db_session: Session) -> None:
    result = UserRepository.get_user_by_username("johndoe", db_session)
    assert result.username == db_user.username
    assert result.email == db_user.email
    assert result.first_name == db_user.first_name
    assert result.last_name == db_user.last_name
    assert result.role == db_user.role


def test_get_user_by_username_fail(db_session: Session) -> None:
    with pytest.raises(RecordNotFoundError) as e:
        UserRepository.get_user_by_username("not_exists", db_session)
    assert e.value.detail == "User not found"


def test_get_user_by_id_success(db_user: User, db_session: Session) -> None:
    result = UserRepository.get_user_by_id(1, db_session)
    assert result.username == db_user.username
    assert result.email == db_user.email
    assert result.first_name == db_user.first_name
    assert result.last_name == db_user.last_name
    assert result.role == db_user.role


def test_get_user_by_id_fail(db_session: Session) -> None:
    with pytest.raises(RecordNotFoundError) as e:
        UserRepository.get_user_by_id(999, db_session)
    assert e.value.detail == "User not found"


def test_get_users_success(db_user: User, db_session: Session) -> None:
    result = UserRepository.get_users(db_session)
    assert len(result) == 1


def test_create_user_success(db_session: Session) -> None:
    user = User(
        username="janedoe",
        email="janedoe@gmail.com",
        first_name="Jane",
        last_name="Doe",
        role="user",
        hashed_password=b"hashed_password",
    )
    UserRepository.create_user(user, db_session)
    db_user = UserRepository.get_user_by_username(user.username, db_session)
    assert user.username == db_user.username
    assert user.email == db_user.email
    assert user.first_name == db_user.first_name
    assert user.last_name == db_user.last_name
    assert user.role == db_user.role


def test_create_user_fail_username_exists(db_user: User, db_session: Session) -> None:
    user = User(
        username="johndoe",
        email="johndoe@gmail.com",
        first_name="John",
        last_name="Doe",
        role="user",
        hashed_password=b"hashed_password",
    )
    with pytest.raises(IntegrityError) as e:
        UserRepository.create_user(user, db_session)
    assert "UNIQUE constraint failed: users.username" in str(e.value)


def test_create_user_fail_email_exists(db_user: User, db_session: Session) -> None:
    user = User(
        username="janedoe",
        email="johndoe@gmail.com",
        first_name="Jane",
        last_name="Doe",
        role="user",
        hashed_password=b"hashed_password",
    )
    with pytest.raises(IntegrityError) as e:
        UserRepository.create_user(user, db_session)
    assert "UNIQUE constraint failed: users.email" in str(e.value)


def test_delete_user_success(db_user: User, db_session: Session) -> None:
    UserRepository.delete_user(db_user, db_session)
    with pytest.raises(RecordNotFoundError) as e:
        UserRepository.get_user_by_id(db_user.id, db_session)
    assert e.value.detail == "User not found"


def test_get_task_by_id_success(db_task: Task, db_session: Session) -> None:
    task = TaskRepository.get_task_by_id(1, db_session)
    assert task.name == db_task.name
    assert task.content == db_task.content
    assert task.brands == db_task.brands
    assert task.probs == db_task.probs


def test_get_task_by_id_fail(db_session: Session) -> None:
    with pytest.raises(RecordNotFoundError) as e:
        TaskRepository.get_task_by_id(999, db_session)
    assert e.value.detail == "Task not found"


def test_get_tasks_for_user_success(db_user: User, db_task: Task, db_session: Session) -> None:
    tasks = TaskRepository.get_tasks_for_user(1, db_session)
    assert len(tasks) == 1
    assert tasks[0].user_id == db_user.id
    assert tasks[0].name == db_task.name
    assert tasks[0].content == db_task.content
    assert tasks[0].brands == db_task.brands
    assert tasks[0].probs == db_task.probs


def test_get_tasks_for_user_fail(db_session: Session) -> None:
    tasks = TaskRepository.get_tasks_for_user(999, db_session)
    assert len(tasks) == 0


def test_create_task_success(db_user: User, db_session: Session) -> None:
    task = Task(
        user_id=db_user.id,
        name="test_task",
        content="test_content",
        brands="BMW",
        probs="0.99",
    )
    TaskRepository.create_task(task, db_session)
    db_task = TaskRepository.get_task_by_id(task.id, db_session)
    assert task.user_id == db_task.user_id
    assert task.name == db_task.name
    assert task.content == db_task.content
    assert task.brands == db_task.brands
    assert task.probs == db_task.probs
