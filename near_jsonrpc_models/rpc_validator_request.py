from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.epoch_id import EpochId
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class RpcValidatorRequestLatest(RootModel[Literal['latest']]):
    pass

class RpcValidatorRequestEpochId(BaseModel):
    epoch_id: EpochId

class RpcValidatorRequestBlockId(BaseModel):
    block_id: BlockId

class RpcValidatorRequest(RootModel[Union[RpcValidatorRequestLatest, RpcValidatorRequestEpochId, RpcValidatorRequestBlockId]]):
    pass

