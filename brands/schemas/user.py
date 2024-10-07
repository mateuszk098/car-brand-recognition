from enum import StrEnum, auto

from pydantic import BaseModel, EmailStr, Field, SecretStr


class Role(StrEnum):
    """Possible user roles."""

    user = auto()
    admin = auto()


class UserBase(BaseModel):
    """Represents base user schema."""

    username: str = Field(min_length=4, max_length=32, description="Unique username.", examples=["john_doe"])
    email: EmailStr = Field(description="Email address.", examples=["johndoe@gmail.com"])
    first_name: str = Field(min_length=2, max_length=32, description="First name.", examples=["John"])
    last_name: str = Field(min_length=2, max_length=64, description="Last name.", examples=["Doe"])
    role: Role = Field(description="User role.")


class UserCreate(UserBase):
    """Represents user creation schema."""

    password: SecretStr = Field(min_length=8, max_length=64, description="User password.", examples=["secret123"])

    class Config:
        json_schema_extra = {
            "example": {
                "username": "john_doe",
                "email": "johndoe@gmail.com",
                "first_name": "John",
                "last_name": "Doe",
                "role": "user",
                "password": "secret123",
            }
        }


class UserSchema(UserBase):
    """Represents user schema in database."""

    id: int
    hashed_password: bytes

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "username": "john_doe",
                "email": "johndoe@gmail.com",
                "first_name": "John",
                "last_name": "Doe",
                "role": "user",
                "hashed_password": "hashed_secret",
            }
        }


class TaskSchema(BaseModel):
    """Represents prediction from deep learning model."""

    id: int
    user_id: int
    name: str
    content: str
    brands: str
    probs: str

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "user_id": 1,
                "name": "image.jpg",
                "content": "image/jpeg",
                "brands": "'Audi', 'BMW', 'Ford'",
                "probs": "0.8, 0.1, 0.05",
            }
        }


class Token(BaseModel):
    """Represents access token schema."""

    access_token: str
    token_type: str
