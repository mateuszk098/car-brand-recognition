"""Repository classes for handling CRUD operations for users and tasks."""

from sqlalchemy.orm import Session

from ..errors import RecordNotFoundError
from .models import Task, User


class UserRepository:
    """Repository for handling CRUD operations for users."""

    @staticmethod
    def get_user_by_username(username: str, db: Session) -> User:
        """Get a user by username."""
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            raise RecordNotFoundError("User")
        return user

    @staticmethod
    def get_user_by_email(email: str, db: Session) -> User:
        """Get a user by email."""
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise RecordNotFoundError("User")
        return user

    @staticmethod
    def get_user_by_id(user_id: int, db: Session) -> User:
        """Get a user by its ID."""
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise RecordNotFoundError("User")
        return user

    @staticmethod
    def get_users(db: Session) -> list[User]:
        """Get all users from database."""
        return db.query(User).all()

    @staticmethod
    def create_user(user: User, db: Session) -> User:
        """Create a new user in database."""
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def delete_user(user: User, db: Session) -> None:
        """Delete a user from database."""
        db.delete(user)
        db.commit()
        return


class TaskRepository:
    """Repository for handling CRUD operations for user's tasks."""

    @staticmethod
    def get_task_by_id(task_id: int, db: Session) -> Task:
        """Retrieve a task by its ID."""
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is None:
            raise RecordNotFoundError("Task")
        return task

    @staticmethod
    def get_tasks_for_user(user_id: int, db: Session) -> list[Task]:
        """Retrieve all tasks for a specific user."""
        return db.query(Task).filter(Task.user_id == user_id).all()

    @staticmethod
    def create_task(task: Task, db: Session) -> Task:
        """Create a new task in database."""
        db.add(task)
        db.commit()
        db.refresh(task)
        return task
