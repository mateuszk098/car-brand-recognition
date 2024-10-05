from ..data import user as data
from ..schemas.user import User


def get_one_user(username: str) -> User | None:
    return data.get_one_user(username)


def get_all_users() -> list[User]:
    return data.get_all_users()


def create(user: User) -> None:
    data.create(user)


def modify(username: str, user: User) -> None:
    data.modify(username, user)


def delete(username: str) -> None:
    data.delete(username)
