from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class RpcClientConfigErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcClientConfigErrorInternalError(BaseModel):
    info: RpcClientConfigErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcClientConfigError(RootModel[Union[RpcClientConfigErrorInternalError]]):
    pass

