from ..schemas.user import User

_users: list[User] = [
    User(username="JohnDoe", email="johndoe@gmail.com"),
    User(username="JaneDoe", email="janedoe@gmail.com"),
]


def get_all_users() -> list[User]:
    return _users


def get_one_user(username: str) -> User | None:
    for user in _users:
        if user.username == username:
            return user


def create(user: User) -> None:
    _users.append(user)


def modify(username: str, user: User) -> None:
    for i, u in enumerate(_users):
        if u.username == username:
            _users[i] = user
            break


def delete(username: str) -> None:
    for i, user in enumerate(_users):
        if user.username == username:
            del _users[i]
            break
