from fastapi import APIRouter

from ..models.user import User
from ..services import user as service

router = APIRouter(prefix="/user", tags=["User"])


@router.get("/{username}", response_model=User)
async def get_one_user(username: str) -> User | None:
    return service.get_one_user(username)


@router.get("/", response_model=list[User])
async def get_all_users() -> list[User]:
    return service.get_all_users()


@router.post("/", response_model=None)
async def create(user: User) -> None:
    service.create(user)


@router.put("/{username}", response_model=None)
async def modify(username: str, user: User) -> None:
    service.modify(username, user)


@router.delete("/{username}", response_model=None)
async def delete(username: str) -> None:
    service.delete(username)
