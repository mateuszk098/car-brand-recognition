"""Pydantic models for user and task schemas."""

from datetime import datetime
from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr


class Role(StrEnum):
    """Possible user roles."""

    user = auto()
    admin = auto()


class Password(BaseModel):
    """Represents password schema."""

    password: SecretStr = Field(min_length=8, max_length=64, description="Password")
    confirm_password: SecretStr = Field(min_length=8, max_length=64, description="Confirm Password")


class UserBase(BaseModel):
    """Represents base user schema."""

    username: str = Field(
        min_length=4,
        max_length=32,
        description="Username",
        examples=["john_doe"],
    )
    email: EmailStr = Field(
        description="Email Address.",
        examples=["johndoe@gmail.com"],
    )
    first_name: str = Field(
        min_length=2,
        max_length=32,
        description="First Name",
        examples=["John"],
    )
    last_name: str = Field(
        min_length=2,
        max_length=64,
        description="Last Name",
        examples=["Doe"],
    )
    role: Role = Field(description="User Role")


class UserCreate(UserBase, Password):
    """Represents user creation schema."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "john_doe",
                "email": "johndoe@gmail.com",
                "first_name": "John",
                "last_name": "Doe",
                "role": "user",
                "password": "secret123",
                "confirm_password": "secret123",
            }
        }
    )


class UserSchema(UserBase):
    """Represents user schema in database."""

    time_created: datetime
    id: int
    hashed_password: bytes

    model_config = ConfigDict(
        from_attributes=True,  # Needed to convert from SQLAlchemy model
        json_schema_extra={
            "example": {
                "time_created": "2021-01-01T00:00:00",
                "id": 1,
                "username": "john_doe",
                "email": "johndoe@gmail.com",
                "first_name": "John",
                "last_name": "Doe",
                "role": "user",
                "hashed_password": "hashed_secret",
            }
        },
    )


class TaskSchema(BaseModel):
    """Represents prediction from deep learning model."""

    time_created: datetime
    id: int
    user_id: int
    name: str
    content: str
    brands: str
    probs: str

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "time_created": "2021-01-01T00:00:00",
                "id": 1,
                "user_id": 1,
                "name": "image.jpg",
                "content": "image/jpeg",
                "brands": "'Audi', 'BMW', 'Ford'",
                "probs": "0.8, 0.1, 0.05",
            }
        },
    )


class Token(BaseModel):
    """Represents access token schema."""

    access_token: str
    token_type: str
