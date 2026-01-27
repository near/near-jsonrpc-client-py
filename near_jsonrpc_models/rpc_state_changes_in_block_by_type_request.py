"""It is a [serializable view] of [`StateChangesRequest`].

[serializable view]: ./index.html
[`StateChangesRequest`]: ../types/struct.StateChangesRequest.html"""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.account_with_public_key import AccountWithPublicKey
from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.finality import Finality
from near_jsonrpc_models.store_key import StoreKey
from near_jsonrpc_models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from typing import List
from typing import Literal
from typing import Union


class RpcStateChangesInBlockByTypeRequestAccountChangesByBlockId(BaseModel):
    block_id: BlockId
    account_ids: List[AccountId]
    changes_type: Literal['account_changes']

class RpcStateChangesInBlockByTypeRequestSingleAccessKeyChangesByBlockId(BaseModel):
    block_id: BlockId
    changes_type: Literal['single_access_key_changes']
    keys: List[AccountWithPublicKey]

class RpcStateChangesInBlockByTypeRequestAllAccessKeyChangesByBlockId(BaseModel):
    block_id: BlockId
    account_ids: List[AccountId]
    changes_type: Literal['all_access_key_changes']

class RpcStateChangesInBlockByTypeRequestContractCodeChangesByBlockId(BaseModel):
    block_id: BlockId
    account_ids: List[AccountId]
    changes_type: Literal['contract_code_changes']

class RpcStateChangesInBlockByTypeRequestDataChangesByBlockId(BaseModel):
    block_id: BlockId
    account_ids: List[AccountId]
    changes_type: Literal['data_changes']
    key_prefix_base64: StoreKey

class RpcStateChangesInBlockByTypeRequestAccountChangesByFinality(BaseModel):
    finality: Finality
    account_ids: List[AccountId]
    changes_type: Literal['account_changes']

class RpcStateChangesInBlockByTypeRequestSingleAccessKeyChangesByFinality(BaseModel):
    finality: Finality
    changes_type: Literal['single_access_key_changes']
    keys: List[AccountWithPublicKey]

class RpcStateChangesInBlockByTypeRequestAllAccessKeyChangesByFinality(BaseModel):
    finality: Finality
    account_ids: List[AccountId]
    changes_type: Literal['all_access_key_changes']

class RpcStateChangesInBlockByTypeRequestContractCodeChangesByFinality(BaseModel):
    finality: Finality
    account_ids: List[AccountId]
    changes_type: Literal['contract_code_changes']

class RpcStateChangesInBlockByTypeRequestDataChangesByFinality(BaseModel):
    finality: Finality
    account_ids: List[AccountId]
    changes_type: Literal['data_changes']
    key_prefix_base64: StoreKey

class RpcStateChangesInBlockByTypeRequestAccountChangesBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    account_ids: List[AccountId]
    changes_type: Literal['account_changes']

class RpcStateChangesInBlockByTypeRequestSingleAccessKeyChangesBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    changes_type: Literal['single_access_key_changes']
    keys: List[AccountWithPublicKey]

class RpcStateChangesInBlockByTypeRequestAllAccessKeyChangesBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    account_ids: List[AccountId]
    changes_type: Literal['all_access_key_changes']

class RpcStateChangesInBlockByTypeRequestContractCodeChangesBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    account_ids: List[AccountId]
    changes_type: Literal['contract_code_changes']

class RpcStateChangesInBlockByTypeRequestDataChangesBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    account_ids: List[AccountId]
    changes_type: Literal['data_changes']
    key_prefix_base64: StoreKey

class RpcStateChangesInBlockByTypeRequest(RootModel[Union[RpcStateChangesInBlockByTypeRequestAccountChangesByBlockId, RpcStateChangesInBlockByTypeRequestSingleAccessKeyChangesByBlockId, RpcStateChangesInBlockByTypeRequestAllAccessKeyChangesByBlockId, RpcStateChangesInBlockByTypeRequestContractCodeChangesByBlockId, RpcStateChangesInBlockByTypeRequestDataChangesByBlockId, RpcStateChangesInBlockByTypeRequestAccountChangesByFinality, RpcStateChangesInBlockByTypeRequestSingleAccessKeyChangesByFinality, RpcStateChangesInBlockByTypeRequestAllAccessKeyChangesByFinality, RpcStateChangesInBlockByTypeRequestContractCodeChangesByFinality, RpcStateChangesInBlockByTypeRequestDataChangesByFinality, RpcStateChangesInBlockByTypeRequestAccountChangesBySyncCheckpoint, RpcStateChangesInBlockByTypeRequestSingleAccessKeyChangesBySyncCheckpoint, RpcStateChangesInBlockByTypeRequestAllAccessKeyChangesBySyncCheckpoint, RpcStateChangesInBlockByTypeRequestContractCodeChangesBySyncCheckpoint, RpcStateChangesInBlockByTypeRequestDataChangesBySyncCheckpoint]]):
    pass

