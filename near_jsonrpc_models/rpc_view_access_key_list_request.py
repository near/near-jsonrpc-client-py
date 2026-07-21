from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.finality import Finality
from near_jsonrpc_models.public_key_handle import PublicKeyHandle
from near_jsonrpc_models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Union


class RpcViewAccessKeyListRequestBlockId(BaseModel):
    account_id: AccountId
    # Pagination cursor: resume the listing strictly after this access key.
    # Pass the `last_key` returned by the previous page.
    after_key: PublicKeyHandle | None = None
    # Maximum number of access keys to return in this page.
    limit: conint(ge=1, le=4294967295) | None = None
    block_id: BlockId

class RpcViewAccessKeyListRequestFinality(BaseModel):
    account_id: AccountId
    # Pagination cursor: resume the listing strictly after this access key.
    # Pass the `last_key` returned by the previous page.
    after_key: PublicKeyHandle | None = None
    # Maximum number of access keys to return in this page.
    limit: conint(ge=1, le=4294967295) | None = None
    finality: Finality

class RpcViewAccessKeyListRequestSyncCheckpoint(BaseModel):
    account_id: AccountId
    # Pagination cursor: resume the listing strictly after this access key.
    # Pass the `last_key` returned by the previous page.
    after_key: PublicKeyHandle | None = None
    # Maximum number of access keys to return in this page.
    limit: conint(ge=1, le=4294967295) | None = None
    sync_checkpoint: SyncCheckpoint

class RpcViewAccessKeyListRequest(RootModel[Union[RpcViewAccessKeyListRequestBlockId, RpcViewAccessKeyListRequestFinality, RpcViewAccessKeyListRequestSyncCheckpoint]]):
    pass

