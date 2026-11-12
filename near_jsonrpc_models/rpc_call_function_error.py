from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.actions_validation_error import ActionsValidationError
from near_jsonrpc_models.block_reference import BlockReference
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.function_call_error import FunctionCallError
from near_jsonrpc_models.global_contract_identifier import GlobalContractIdentifier
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class RpcCallFunctionErrorUnknownBlockInfo(BaseModel):
    block_reference: BlockReference

class RpcCallFunctionErrorUnknownBlock(BaseModel):
    info: RpcCallFunctionErrorUnknownBlockInfo
    name: Literal['UNKNOWN_BLOCK']

class RpcCallFunctionErrorInvalidAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcCallFunctionErrorInvalidAccount(BaseModel):
    info: RpcCallFunctionErrorInvalidAccountInfo
    name: Literal['INVALID_ACCOUNT']

class RpcCallFunctionErrorUnknownAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcCallFunctionErrorUnknownAccount(BaseModel):
    info: RpcCallFunctionErrorUnknownAccountInfo
    name: Literal['UNKNOWN_ACCOUNT']

class RpcCallFunctionErrorUninitializedAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcCallFunctionErrorUninitializedAccount(BaseModel):
    info: RpcCallFunctionErrorUninitializedAccountInfo
    name: Literal['UNINITIALIZED_ACCOUNT']

class RpcCallFunctionErrorInvalidStateInitInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    error: ActionsValidationError

class RpcCallFunctionErrorInvalidStateInit(BaseModel):
    info: RpcCallFunctionErrorInvalidStateInitInfo
    name: Literal['INVALID_STATE_INIT']

class RpcCallFunctionErrorNoContractCodeInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    contract_account_id: AccountId

class RpcCallFunctionErrorNoContractCode(BaseModel):
    info: RpcCallFunctionErrorNoContractCodeInfo
    name: Literal['NO_CONTRACT_CODE']

class RpcCallFunctionErrorNoGlobalContractCodeInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    identifier: GlobalContractIdentifier

class RpcCallFunctionErrorNoGlobalContractCode(BaseModel):
    info: RpcCallFunctionErrorNoGlobalContractCodeInfo
    name: Literal['NO_GLOBAL_CONTRACT_CODE']

class RpcCallFunctionErrorContractExecutionErrorInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    vm_error: FunctionCallError

class RpcCallFunctionErrorContractExecutionError(BaseModel):
    info: RpcCallFunctionErrorContractExecutionErrorInfo
    name: Literal['CONTRACT_EXECUTION_ERROR']

class RpcCallFunctionErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcCallFunctionErrorInternalError(BaseModel):
    info: RpcCallFunctionErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcCallFunctionError(RootModel[Union[RpcCallFunctionErrorUnknownBlock, RpcCallFunctionErrorInvalidAccount, RpcCallFunctionErrorUnknownAccount, RpcCallFunctionErrorUninitializedAccount, RpcCallFunctionErrorInvalidStateInit, RpcCallFunctionErrorNoContractCode, RpcCallFunctionErrorNoGlobalContractCode, RpcCallFunctionErrorContractExecutionError, RpcCallFunctionErrorInternalError]]):
    pass

