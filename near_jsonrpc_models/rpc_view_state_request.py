from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.finality import Finality
from near_jsonrpc_models.store_key import StoreKey
from near_jsonrpc_models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Union


class RpcViewStateRequestBlockId(BaseModel):
    account_id: AccountId
    after_key_base64: StoreKey | None = None
    include_proof: bool = False
    limit: conint(ge=1, le=4294967295) | None = None
    prefix_base64: StoreKey
    block_id: BlockId

class RpcViewStateRequestFinality(BaseModel):
    account_id: AccountId
    after_key_base64: StoreKey | None = None
    include_proof: bool = False
    limit: conint(ge=1, le=4294967295) | None = None
    prefix_base64: StoreKey
    finality: Finality

class RpcViewStateRequestSyncCheckpoint(BaseModel):
    account_id: AccountId
    after_key_base64: StoreKey | None = None
    include_proof: bool = False
    limit: conint(ge=1, le=4294967295) | None = None
    prefix_base64: StoreKey
    sync_checkpoint: SyncCheckpoint

class RpcViewStateRequest(RootModel[Union[RpcViewStateRequestBlockId, RpcViewStateRequestFinality, RpcViewStateRequestSyncCheckpoint]]):
    pass

