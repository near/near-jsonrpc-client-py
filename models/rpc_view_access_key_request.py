from models.account_id import AccountId
from models.block_id import BlockId
from models.finality import Finality
from models.public_key import PublicKey
from models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcViewAccessKeyRequestBlockId(BaseModel):
    account_id: AccountId
    public_key: PublicKey
    block_id: BlockId

class RpcViewAccessKeyRequestFinality(BaseModel):
    account_id: AccountId
    public_key: PublicKey
    finality: Finality

class RpcViewAccessKeyRequestSyncCheckpoint(BaseModel):
    account_id: AccountId
    public_key: PublicKey
    sync_checkpoint: SyncCheckpoint

class RpcViewAccessKeyRequest(RootModel[Union[RpcViewAccessKeyRequestBlockId, RpcViewAccessKeyRequestFinality, RpcViewAccessKeyRequestSyncCheckpoint]]):
    pass

