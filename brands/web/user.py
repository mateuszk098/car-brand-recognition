from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ..data import get_db
from ..errors.user import MissingUserError, UserAlreadyExistsError
from ..schemas.user import TaskCreate, TaskSchema, UserCreate, UserSchema
from ..services import user as service

router = APIRouter(prefix="/user", tags=["User"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/token")


@router.get("/{username}", status_code=status.HTTP_200_OK, response_model=UserSchema)
async def get_user(username: str, db: Session = Depends(get_db)) -> UserSchema:
    try:
        return service.get_user(username, db)
    except MissingUserError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"User {e} not found")


@router.get("/", status_code=status.HTTP_200_OK, response_model=list[UserSchema])
async def get_users(db: Session = Depends(get_db)) -> list[UserSchema]:
    return service.get_users(db)


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserSchema)
async def register(user: UserCreate, db: Session = Depends(get_db)) -> UserSchema:
    try:
        return service.create_user(user, db)
    except UserAlreadyExistsError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"User {e} already exists")


@router.post("/predict", status_code=status.HTTP_201_CREATED, response_model=TaskSchema)
async def predict(task: TaskCreate, db: Session = Depends(get_db)) -> TaskSchema:
    return service.create_task(task, 1, db)


@router.get("/tasks/", status_code=status.HTTP_200_OK, response_model=list[TaskSchema])
async def get_tasks(db: Session = Depends(get_db)) -> list[TaskSchema]:
    return service.get_tasks(1, db)
