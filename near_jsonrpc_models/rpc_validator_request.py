from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.epoch_id import EpochId
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcValidatorRequestEpochId(BaseModel):
    epoch_id: EpochId

class RpcValidatorRequestBlockId(BaseModel):
    block_id: BlockId

class RpcValidatorRequestLatest(BaseModel):
    latest: str

class RpcValidatorRequest(RootModel[Union[RpcValidatorRequestEpochId, RpcValidatorRequestBlockId, RpcValidatorRequestLatest]]):
    pass

