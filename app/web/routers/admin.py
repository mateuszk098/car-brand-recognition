from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, status

from app.errors import ForbiddenHTTPError, InvalidPasswordError, PasswordMismatchError, UserNotFoundError
from app.schema.user import Password, Role, UserSchema
from app.service import user as service

from ..dependencies import DBDep, UserDep

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get(
    "/users/{username}",
    response_model=UserSchema,
    status_code=status.HTTP_200_OK,
    summary="Get User by Username",
)
async def get_user_by_username(username: str, user: UserDep, db: DBDep) -> UserSchema:
    if user.role != Role.admin:
        raise ForbiddenHTTPError()
    try:
        return service.get_user_by_username(username, db)
    except UserNotFoundError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=e.detail)


@router.get(
    "/users/",
    response_model=list[UserSchema],
    status_code=status.HTTP_200_OK,
    summary="Get All Users",
)
async def get_users(user: UserDep, db: DBDep) -> list[UserSchema]:
    if user.role != Role.admin:
        raise ForbiddenHTTPError()
    return service.get_users(db)


@router.delete(
    "/users/{username}",
    response_model=None,
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete User by Username",
)
async def delete_user(
    username: str,
    password: Annotated[Password, Form(description="Admin password")],
    user: UserDep,
    db: DBDep,
) -> None:
    if user.role != Role.admin:
        raise ForbiddenHTTPError()
    try:
        service.delete_user_by_username(username, password, user, db)
    except PasswordMismatchError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=e.detail)
    except InvalidPasswordError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=e.detail)
    except UserNotFoundError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=e.detail)
