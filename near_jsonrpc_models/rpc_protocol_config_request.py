from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.finality import Finality
from near_jsonrpc_models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcProtocolConfigRequestBlockId(BaseModel):
    block_id: BlockId

class RpcProtocolConfigRequestFinality(BaseModel):
    finality: Finality

class RpcProtocolConfigRequestSyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint

class RpcProtocolConfigRequest(RootModel[Union[RpcProtocolConfigRequestBlockId, RpcProtocolConfigRequestFinality, RpcProtocolConfigRequestSyncCheckpoint]]):
    pass

