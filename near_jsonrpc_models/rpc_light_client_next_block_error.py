from near_jsonrpc_models.epoch_id import EpochId
from pydantic import BaseModel
from pydantic import RootModel
from typing import Any
from typing import Dict
from typing import Literal
from typing import Union


class RpcLightClientNextBlockErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcLightClientNextBlockErrorInternalError(BaseModel):
    info: RpcLightClientNextBlockErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcLightClientNextBlockErrorUnknownBlock(BaseModel):
    info: Dict[str, Any]
    name: Literal['UNKNOWN_BLOCK']

class RpcLightClientNextBlockErrorEpochOutOfBoundsInfo(BaseModel):
    epoch_id: EpochId

class RpcLightClientNextBlockErrorEpochOutOfBounds(BaseModel):
    info: RpcLightClientNextBlockErrorEpochOutOfBoundsInfo
    name: Literal['EPOCH_OUT_OF_BOUNDS']

class RpcLightClientNextBlockError(RootModel[Union[RpcLightClientNextBlockErrorInternalError, RpcLightClientNextBlockErrorUnknownBlock, RpcLightClientNextBlockErrorEpochOutOfBounds]]):
    pass

