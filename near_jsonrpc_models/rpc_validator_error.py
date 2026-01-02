from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class RpcValidatorErrorUnknownEpoch(BaseModel):
    name: Literal['UNKNOWN_EPOCH']

class RpcValidatorErrorValidatorInfoUnavailable(BaseModel):
    name: Literal['VALIDATOR_INFO_UNAVAILABLE']

class RpcValidatorErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcValidatorErrorInternalError(BaseModel):
    info: RpcValidatorErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcValidatorError(RootModel[Union[RpcValidatorErrorUnknownEpoch, RpcValidatorErrorValidatorInfoUnavailable, RpcValidatorErrorInternalError]]):
    pass

