from pydantic import BaseModel
from pydantic import RootModel
from typing import Any
from typing import Dict
from typing import Literal
from typing import Union


class RpcStateChangesErrorUnknownBlock(BaseModel):
    info: Dict[str, Any]
    name: Literal['UNKNOWN_BLOCK']

class RpcStateChangesErrorNotSyncedYet(BaseModel):
    name: Literal['NOT_SYNCED_YET']

class RpcStateChangesErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcStateChangesErrorInternalError(BaseModel):
    info: RpcStateChangesErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcStateChangesError(RootModel[Union[RpcStateChangesErrorUnknownBlock, RpcStateChangesErrorNotSyncedYet, RpcStateChangesErrorInternalError]]):
    pass

