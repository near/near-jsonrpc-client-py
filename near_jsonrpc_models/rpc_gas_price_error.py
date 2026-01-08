from pydantic import BaseModel
from pydantic import RootModel
from typing import Any
from typing import Dict
from typing import Literal
from typing import Union


class RpcGasPriceErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcGasPriceErrorInternalError(BaseModel):
    info: RpcGasPriceErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcGasPriceErrorUnknownBlock(BaseModel):
    info: Dict[str, Any]
    name: Literal['UNKNOWN_BLOCK']

class RpcGasPriceError(RootModel[Union[RpcGasPriceErrorInternalError, RpcGasPriceErrorUnknownBlock]]):
    pass

