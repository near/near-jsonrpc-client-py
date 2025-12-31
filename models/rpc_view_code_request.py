from models.account_id import AccountId
from models.block_id import BlockId
from models.finality import Finality
from models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcViewCodeRequestBlockId(BaseModel):
    account_id: AccountId
    block_id: BlockId

class RpcViewCodeRequestFinality(BaseModel):
    account_id: AccountId
    finality: Finality

class RpcViewCodeRequestSyncCheckpoint(BaseModel):
    account_id: AccountId
    sync_checkpoint: SyncCheckpoint

class RpcViewCodeRequest(RootModel[Union[RpcViewCodeRequestBlockId, RpcViewCodeRequestFinality, RpcViewCodeRequestSyncCheckpoint]]):
    pass

