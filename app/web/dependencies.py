from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from resnet.network.arch import SEResNet

from ..data import get_db
from ..schema.user import UserSchema
from ..service.auth import get_current_user
from ..service.utils import load_model, oauth2_bearer

DBDep = Annotated[Session, Depends(get_db)]
LoginDep = Annotated[OAuth2PasswordRequestForm, Depends()]
TokenDep = Annotated[str, Depends(oauth2_bearer)]
UserDep = Annotated[UserSchema, Depends(get_current_user)]
ModelDep = Annotated[SEResNet, Depends(load_model, use_cache=False)]  # Model is already cached.
