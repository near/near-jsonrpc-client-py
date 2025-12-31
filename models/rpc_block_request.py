from models.block_id import BlockId
from models.finality import Finality
from models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcBlockRequestBlockId(BaseModel):
    block_id: BlockId

class RpcBlockRequestFinality(BaseModel):
    finality: Finality

class RpcBlockRequestSyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint

class RpcBlockRequest(RootModel[Union[RpcBlockRequestBlockId, RpcBlockRequestFinality, RpcBlockRequestSyncCheckpoint]]):
    pass

