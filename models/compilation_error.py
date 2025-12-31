from models.account_id import AccountId
from models.prepare_error import PrepareError
from models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class CompilationErrorCodeDoesNotExistPayload(BaseModel):
    account_id: AccountId

class CompilationErrorCodeDoesNotExist(StrictBaseModel):
    CodeDoesNotExist: CompilationErrorCodeDoesNotExistPayload

class CompilationErrorPrepareError(StrictBaseModel):
    PrepareError: PrepareError

class CompilationErrorWasmerCompileErrorPayload(BaseModel):
    msg: str

class CompilationErrorWasmerCompileError(StrictBaseModel):
    """This is for defense in depth.
We expect our runtime-independent preparation code to fully catch all invalid wasms,
but, if it ever misses something we’ll emit this error"""
    WasmerCompileError: CompilationErrorWasmerCompileErrorPayload

class CompilationError(RootModel[Union[CompilationErrorCodeDoesNotExist, CompilationErrorPrepareError, CompilationErrorWasmerCompileError]]):
    pass

