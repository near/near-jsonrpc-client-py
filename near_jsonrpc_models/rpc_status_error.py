from near_jsonrpc_models.epoch_id import EpochId
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from pydantic import conlist
from typing import List
from typing import Literal
from typing import Union


class RpcStatusErrorNodeIsSyncing(BaseModel):
    name: Literal['NODE_IS_SYNCING']

class RpcStatusErrorNoNewBlocksInfo(BaseModel):
    elapsed: conlist(conint(ge=0, le=18446744073709551615), min_length=2, max_length=2)

class RpcStatusErrorNoNewBlocks(BaseModel):
    info: RpcStatusErrorNoNewBlocksInfo
    name: Literal['NO_NEW_BLOCKS']

class RpcStatusErrorEpochOutOfBoundsInfo(BaseModel):
    epoch_id: EpochId

class RpcStatusErrorEpochOutOfBounds(BaseModel):
    info: RpcStatusErrorEpochOutOfBoundsInfo
    name: Literal['EPOCH_OUT_OF_BOUNDS']

class RpcStatusErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcStatusErrorInternalError(BaseModel):
    info: RpcStatusErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcStatusError(RootModel[Union[RpcStatusErrorNodeIsSyncing, RpcStatusErrorNoNewBlocks, RpcStatusErrorEpochOutOfBounds, RpcStatusErrorInternalError]]):
    pass

