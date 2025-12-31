from models.account_id import AccountId
from models.block_id import BlockId
from models.finality import Finality
from models.store_key import StoreKey
from models.sync_checkpoint import SyncCheckpoint
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

