from pydantic import BaseModel


class TaskBase(BaseModel):
    prediction: str
    probability: float


class TaskCreate(TaskBase):
    pass

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "Audi",
                "probability": 0.99,
            }
        }


class TaskSchema(TaskBase):
    id: int
    user_id: int

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "user_id": 1,
                "prediction": "Audi",
                "probability": 0.99,
            }
        }


class UserBase(BaseModel):
    username: str
    email: str
    first_name: str
    last_name: str
    role: str


class UserCreate(UserBase):
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "username": "john_doe",
                "email": "johndoe@gmail.com",
                "first_name": "John",
                "last_name": "Doe",
                "role": "user",
                "password": "secret",
            }
        }


class UserSchema(UserBase):
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


class Token(BaseModel):
    """Access token schema."""

    access_token: str
    token_type: str
