from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from resnet.network.arch import SEResNet

from ..data import get_db
from ..schemas.user import UserSchema
from ..services import user as service
from ..services.classifier import load_model

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="/user/token")

DBDep = Annotated[Session, Depends(get_db)]
LoginDep = Annotated[OAuth2PasswordRequestForm, Depends()]
TokenDep = Annotated[str, Depends(oauth2_bearer)]
ModelDep = Annotated[SEResNet, Depends(load_model, use_cache=False)]  # Model is already cached.


async def get_current_user(token: TokenDep, db: DBDep) -> UserSchema:
    username = service.extract_username_from_token(token)
    return service.get_user(username, db)


UserDep = Annotated[UserSchema, Depends(get_current_user)]
