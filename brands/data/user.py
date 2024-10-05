from ..fake import user as fake
from ..schemas.user import User


def get_all_users() -> list[User]:
    return fake.get_all_users()


def get_one_user(username: str) -> User | None:
    return fake.get_one_user(username)


def create(user: User) -> None:
    fake.create(user)


def modify(username: str, user: User) -> None:
    fake.modify(username, user)


def delete(username: str) -> None:
    fake.delete(username)
