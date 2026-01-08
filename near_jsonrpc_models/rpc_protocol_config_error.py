from pydantic import BaseModel
from pydantic import RootModel
from typing import Any
from typing import Dict
from typing import Literal
from typing import Union


class RpcProtocolConfigErrorUnknownBlock(BaseModel):
    info: Dict[str, Any]
    name: Literal['UNKNOWN_BLOCK']

class RpcProtocolConfigErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcProtocolConfigErrorInternalError(BaseModel):
    info: RpcProtocolConfigErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcProtocolConfigError(RootModel[Union[RpcProtocolConfigErrorUnknownBlock, RpcProtocolConfigErrorInternalError]]):
    pass

