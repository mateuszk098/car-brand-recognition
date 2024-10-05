from pydantic import BaseModel


class User(BaseModel):
    username: str
    email: str


class UserIn(User):
    password: str


class UserOut(User):
    hashed_password: bytes
