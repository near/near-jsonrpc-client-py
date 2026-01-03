from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class RpcNetworkInfoErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcNetworkInfoErrorInternalError(BaseModel):
    info: RpcNetworkInfoErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcNetworkInfoError(RootModel[Union[RpcNetworkInfoErrorInternalError]]):
    pass

