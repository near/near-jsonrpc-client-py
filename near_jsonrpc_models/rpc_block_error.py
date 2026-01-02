from pydantic import BaseModel
from pydantic import RootModel
from typing import Any
from typing import Dict
from typing import Literal
from typing import Union


class RpcBlockErrorUnknownBlock(BaseModel):
    info: Dict[str, Any]
    name: Literal['UNKNOWN_BLOCK']

class RpcBlockErrorNotSyncedYet(BaseModel):
    name: Literal['NOT_SYNCED_YET']

class RpcBlockErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcBlockErrorInternalError(BaseModel):
    info: RpcBlockErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcBlockError(RootModel[Union[RpcBlockErrorUnknownBlock, RpcBlockErrorNotSyncedYet, RpcBlockErrorInternalError]]):
    pass

