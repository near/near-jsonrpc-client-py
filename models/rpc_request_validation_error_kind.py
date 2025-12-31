from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class RpcRequestValidationErrorKindMethodNotFoundInfo(BaseModel):
    method_name: str

class RpcRequestValidationErrorKindMethodNotFound(BaseModel):
    info: RpcRequestValidationErrorKindMethodNotFoundInfo
    name: Literal['METHOD_NOT_FOUND']

class RpcRequestValidationErrorKindParseErrorInfo(BaseModel):
    error_message: str

class RpcRequestValidationErrorKindParseError(BaseModel):
    info: RpcRequestValidationErrorKindParseErrorInfo
    name: Literal['PARSE_ERROR']

class RpcRequestValidationErrorKind(RootModel[Union[RpcRequestValidationErrorKindMethodNotFound, RpcRequestValidationErrorKindParseError]]):
    pass

