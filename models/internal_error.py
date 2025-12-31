from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class InternalErrorInternalErrorInfo(BaseModel):
    error_message: str

class InternalErrorInternalError(BaseModel):
    info: InternalErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class InternalError(RootModel[Union[InternalErrorInternalError]]):
    pass

