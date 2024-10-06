# from fastapi import APIRouter, Depends, HTTPException, status
# from sqlalchemy.orm import Session

# from ..data import get_db
# from ..errors.user import MissingUserError
# from ..schemas.user import UserSchema
# from ..services import admin as service

# router = APIRouter(prefix="/admin", tags=["Admin"])


# @router.get("/{username}", status_code=status.HTTP_200_OK, response_model=UserSchema)
# async def get_user(username: str, db: Session = Depends(get_db)) -> UserSchema:
#     try:
#         return service.get_user(username, db)
#     except MissingUserError as e:
#         raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"User {e} not found")


# @router.get("/", status_code=status.HTTP_200_OK, response_model=list[UserSchema])
# async def get_users(db: Session = Depends(get_db)) -> list[UserSchema]:
#     return service.get_users(db)
