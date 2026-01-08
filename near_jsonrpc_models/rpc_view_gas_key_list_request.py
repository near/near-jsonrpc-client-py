from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.finality import Finality
from near_jsonrpc_models.sync_checkpoint import SyncCheckpoint
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

