from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.finality import Finality
from near_jsonrpc_models.function_args import FunctionArgs
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.store_key import StoreKey
from near_jsonrpc_models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class RpcQueryRequestViewAccountByBlockId(BaseModel):
    block_id: BlockId
    account_id: AccountId
    request_type: Literal['view_account']

class RpcQueryRequestViewCodeByBlockId(BaseModel):
    block_id: BlockId
    account_id: AccountId
    request_type: Literal['view_code']

class RpcQueryRequestViewStateByBlockId(BaseModel):
    block_id: BlockId
    account_id: AccountId
    include_proof: bool = None
    prefix_base64: StoreKey
    request_type: Literal['view_state']

class RpcQueryRequestViewAccessKeyByBlockId(BaseModel):
    block_id: BlockId
    account_id: AccountId
    public_key: PublicKey
    request_type: Literal['view_access_key']

class RpcQueryRequestViewAccessKeyListByBlockId(BaseModel):
    block_id: BlockId
    account_id: AccountId
    request_type: Literal['view_access_key_list']

class RpcQueryRequestViewGasKeyNoncesByBlockId(BaseModel):
    block_id: BlockId
    account_id: AccountId
    public_key: PublicKey
    request_type: Literal['view_gas_key_nonces']

class RpcQueryRequestCallFunctionByBlockId(BaseModel):
    block_id: BlockId
    account_id: AccountId
    args_base64: FunctionArgs
    method_name: str
    request_type: Literal['call_function']

class RpcQueryRequestViewGlobalContractCodeByBlockId(BaseModel):
    block_id: BlockId
    code_hash: CryptoHash
    request_type: Literal['view_global_contract_code']

class RpcQueryRequestViewGlobalContractCodeByAccountIdByBlockId(BaseModel):
    block_id: BlockId
    account_id: AccountId
    request_type: Literal['view_global_contract_code_by_account_id']

class RpcQueryRequestViewAccountByFinality(BaseModel):
    finality: Finality
    account_id: AccountId
    request_type: Literal['view_account']

class RpcQueryRequestViewCodeByFinality(BaseModel):
    finality: Finality
    account_id: AccountId
    request_type: Literal['view_code']

class RpcQueryRequestViewStateByFinality(BaseModel):
    finality: Finality
    account_id: AccountId
    include_proof: bool = None
    prefix_base64: StoreKey
    request_type: Literal['view_state']

class RpcQueryRequestViewAccessKeyByFinality(BaseModel):
    finality: Finality
    account_id: AccountId
    public_key: PublicKey
    request_type: Literal['view_access_key']

class RpcQueryRequestViewAccessKeyListByFinality(BaseModel):
    finality: Finality
    account_id: AccountId
    request_type: Literal['view_access_key_list']

class RpcQueryRequestViewGasKeyNoncesByFinality(BaseModel):
    finality: Finality
    account_id: AccountId
    public_key: PublicKey
    request_type: Literal['view_gas_key_nonces']

class RpcQueryRequestCallFunctionByFinality(BaseModel):
    finality: Finality
    account_id: AccountId
    args_base64: FunctionArgs
    method_name: str
    request_type: Literal['call_function']

class RpcQueryRequestViewGlobalContractCodeByFinality(BaseModel):
    finality: Finality
    code_hash: CryptoHash
    request_type: Literal['view_global_contract_code']

class RpcQueryRequestViewGlobalContractCodeByAccountIdByFinality(BaseModel):
    finality: Finality
    account_id: AccountId
    request_type: Literal['view_global_contract_code_by_account_id']

class RpcQueryRequestViewAccountBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    account_id: AccountId
    request_type: Literal['view_account']

class RpcQueryRequestViewCodeBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    account_id: AccountId
    request_type: Literal['view_code']

class RpcQueryRequestViewStateBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    account_id: AccountId
    include_proof: bool = None
    prefix_base64: StoreKey
    request_type: Literal['view_state']

class RpcQueryRequestViewAccessKeyBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    account_id: AccountId
    public_key: PublicKey
    request_type: Literal['view_access_key']

class RpcQueryRequestViewAccessKeyListBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    account_id: AccountId
    request_type: Literal['view_access_key_list']

class RpcQueryRequestViewGasKeyNoncesBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    account_id: AccountId
    public_key: PublicKey
    request_type: Literal['view_gas_key_nonces']

class RpcQueryRequestCallFunctionBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    account_id: AccountId
    args_base64: FunctionArgs
    method_name: str
    request_type: Literal['call_function']

class RpcQueryRequestViewGlobalContractCodeBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    code_hash: CryptoHash
    request_type: Literal['view_global_contract_code']

class RpcQueryRequestViewGlobalContractCodeByAccountIdBySyncCheckpoint(BaseModel):
    sync_checkpoint: SyncCheckpoint
    account_id: AccountId
    request_type: Literal['view_global_contract_code_by_account_id']

class RpcQueryRequest(RootModel[Union[RpcQueryRequestViewAccountByBlockId, RpcQueryRequestViewCodeByBlockId, RpcQueryRequestViewStateByBlockId, RpcQueryRequestViewAccessKeyByBlockId, RpcQueryRequestViewAccessKeyListByBlockId, RpcQueryRequestViewGasKeyNoncesByBlockId, RpcQueryRequestCallFunctionByBlockId, RpcQueryRequestViewGlobalContractCodeByBlockId, RpcQueryRequestViewGlobalContractCodeByAccountIdByBlockId, RpcQueryRequestViewAccountByFinality, RpcQueryRequestViewCodeByFinality, RpcQueryRequestViewStateByFinality, RpcQueryRequestViewAccessKeyByFinality, RpcQueryRequestViewAccessKeyListByFinality, RpcQueryRequestViewGasKeyNoncesByFinality, RpcQueryRequestCallFunctionByFinality, RpcQueryRequestViewGlobalContractCodeByFinality, RpcQueryRequestViewGlobalContractCodeByAccountIdByFinality, RpcQueryRequestViewAccountBySyncCheckpoint, RpcQueryRequestViewCodeBySyncCheckpoint, RpcQueryRequestViewStateBySyncCheckpoint, RpcQueryRequestViewAccessKeyBySyncCheckpoint, RpcQueryRequestViewAccessKeyListBySyncCheckpoint, RpcQueryRequestViewGasKeyNoncesBySyncCheckpoint, RpcQueryRequestCallFunctionBySyncCheckpoint, RpcQueryRequestViewGlobalContractCodeBySyncCheckpoint, RpcQueryRequestViewGlobalContractCodeByAccountIdBySyncCheckpoint]]):
    pass

