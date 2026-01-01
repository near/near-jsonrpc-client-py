from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.finality import Finality
from near_jsonrpc_models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcViewAccessKeyListRequestBlockId(BaseModel):
    account_id: AccountId
    block_id: BlockId

class RpcViewAccessKeyListRequestFinality(BaseModel):
    account_id: AccountId
    finality: Finality

class RpcViewAccessKeyListRequestSyncCheckpoint(BaseModel):
    account_id: AccountId
    sync_checkpoint: SyncCheckpoint

class RpcViewAccessKeyListRequest(RootModel[Union[RpcViewAccessKeyListRequestBlockId, RpcViewAccessKeyListRequestFinality, RpcViewAccessKeyListRequestSyncCheckpoint]]):
    pass

