from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.finality import Finality
from near_jsonrpc_models.store_key import StoreKey
from near_jsonrpc_models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcViewStateRequestBlockId(BaseModel):
    account_id: AccountId
    include_proof: bool = False
    prefix_base64: StoreKey
    block_id: BlockId

class RpcViewStateRequestFinality(BaseModel):
    account_id: AccountId
    include_proof: bool = False
    prefix_base64: StoreKey
    finality: Finality

class RpcViewStateRequestSyncCheckpoint(BaseModel):
    account_id: AccountId
    include_proof: bool = False
    prefix_base64: StoreKey
    sync_checkpoint: SyncCheckpoint

class RpcViewStateRequest(RootModel[Union[RpcViewStateRequestBlockId, RpcViewStateRequestFinality, RpcViewStateRequestSyncCheckpoint]]):
    pass

