from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..data import get_db
from ..schemas.user import TokenMetadata
from ..services import user as service

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="/user/token")


async def get_current_user_data(token: Annotated[str, Depends(oauth2_bearer)]) -> TokenMetadata:
    return service.get_token_metadata(token)


DBDep = Annotated[Session, Depends(get_db)]
TokenDep = Annotated[TokenMetadata, Depends(get_current_user_data)]
LoginDep = Annotated[OAuth2PasswordRequestForm, Depends()]
