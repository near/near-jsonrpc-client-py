from models.account_id import AccountId
from models.block_id import BlockId
from models.finality import Finality
from models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcViewGasKeyListRequestBlockId(BaseModel):
    account_id: AccountId
    block_id: BlockId

class RpcViewGasKeyListRequestFinality(BaseModel):
    account_id: AccountId
    finality: Finality

class RpcViewGasKeyListRequestSyncCheckpoint(BaseModel):
    account_id: AccountId
    sync_checkpoint: SyncCheckpoint

class RpcViewGasKeyListRequest(RootModel[Union[RpcViewGasKeyListRequestBlockId, RpcViewGasKeyListRequestFinality, RpcViewGasKeyListRequestSyncCheckpoint]]):
    pass

