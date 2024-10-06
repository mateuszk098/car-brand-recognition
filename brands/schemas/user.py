from pydantic import BaseModel


class TaskBase(BaseModel):
    prediction: str
    probability: float


class TaskCreate(TaskBase):
    pass


class TaskSchema(TaskBase):
    id: int
    user_id: int

    class Config:
        from_attributes = True


class UserBase(BaseModel):
    username: str
    email: str
    first_name: str
    last_name: str
    role: str


class UserCreate(UserBase):
    password: str


class UserSchema(UserBase):
    id: int
    hashed_password: bytes

    class Config:
        from_attributes = True
