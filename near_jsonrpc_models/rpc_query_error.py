from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_reference import BlockReference
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.function_call_error import FunctionCallError
from near_jsonrpc_models.global_contract_identifier import GlobalContractIdentifier
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class RpcQueryErrorNoSyncedBlocks(BaseModel):
    name: Literal['NO_SYNCED_BLOCKS']

class RpcQueryErrorUnavailableShardInfo(BaseModel):
    requested_shard_id: ShardId

class RpcQueryErrorUnavailableShard(BaseModel):
    info: RpcQueryErrorUnavailableShardInfo
    name: Literal['UNAVAILABLE_SHARD']

class RpcQueryErrorGarbageCollectedBlockInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)

class RpcQueryErrorGarbageCollectedBlock(BaseModel):
    info: RpcQueryErrorGarbageCollectedBlockInfo
    name: Literal['GARBAGE_COLLECTED_BLOCK']

class RpcQueryErrorUnknownBlockInfo(BaseModel):
    block_reference: BlockReference

class RpcQueryErrorUnknownBlock(BaseModel):
    info: RpcQueryErrorUnknownBlockInfo
    name: Literal['UNKNOWN_BLOCK']

class RpcQueryErrorInvalidAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcQueryErrorInvalidAccount(BaseModel):
    info: RpcQueryErrorInvalidAccountInfo
    name: Literal['INVALID_ACCOUNT']

class RpcQueryErrorUnknownAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcQueryErrorUnknownAccount(BaseModel):
    info: RpcQueryErrorUnknownAccountInfo
    name: Literal['UNKNOWN_ACCOUNT']

class RpcQueryErrorNoContractCodeInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    contract_account_id: AccountId

class RpcQueryErrorNoContractCode(BaseModel):
    info: RpcQueryErrorNoContractCodeInfo
    name: Literal['NO_CONTRACT_CODE']

class RpcQueryErrorTooLargeContractStateInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    contract_account_id: AccountId

class RpcQueryErrorTooLargeContractState(BaseModel):
    info: RpcQueryErrorTooLargeContractStateInfo
    name: Literal['TOO_LARGE_CONTRACT_STATE']

class RpcQueryErrorUnknownAccessKeyInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    public_key: PublicKey

class RpcQueryErrorUnknownAccessKey(BaseModel):
    info: RpcQueryErrorUnknownAccessKeyInfo
    name: Literal['UNKNOWN_ACCESS_KEY']

class RpcQueryErrorUnknownGasKeyInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    public_key: PublicKey

class RpcQueryErrorUnknownGasKey(BaseModel):
    info: RpcQueryErrorUnknownGasKeyInfo
    name: Literal['UNKNOWN_GAS_KEY']

class RpcQueryErrorContractExecutionErrorInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    error: FunctionCallError
    vm_error: str

class RpcQueryErrorContractExecutionError(BaseModel):
    info: RpcQueryErrorContractExecutionErrorInfo
    name: Literal['CONTRACT_EXECUTION_ERROR']

class RpcQueryErrorNoGlobalContractCodeInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    identifier: GlobalContractIdentifier

class RpcQueryErrorNoGlobalContractCode(BaseModel):
    info: RpcQueryErrorNoGlobalContractCodeInfo
    name: Literal['NO_GLOBAL_CONTRACT_CODE']

class RpcQueryErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcQueryErrorInternalError(BaseModel):
    info: RpcQueryErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcQueryError(RootModel[Union[RpcQueryErrorNoSyncedBlocks, RpcQueryErrorUnavailableShard, RpcQueryErrorGarbageCollectedBlock, RpcQueryErrorUnknownBlock, RpcQueryErrorInvalidAccount, RpcQueryErrorUnknownAccount, RpcQueryErrorNoContractCode, RpcQueryErrorTooLargeContractState, RpcQueryErrorUnknownAccessKey, RpcQueryErrorUnknownGasKey, RpcQueryErrorContractExecutionError, RpcQueryErrorNoGlobalContractCode, RpcQueryErrorInternalError]]):
    pass

