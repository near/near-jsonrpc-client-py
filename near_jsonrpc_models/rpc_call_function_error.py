from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_reference import BlockReference
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.function_call_error import FunctionCallError
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

class RpcCallFunctionErrorNoContractCodeInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    contract_account_id: AccountId

class RpcCallFunctionErrorNoContractCode(BaseModel):
    info: RpcCallFunctionErrorNoContractCodeInfo
    name: Literal['NO_CONTRACT_CODE']

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

class RpcCallFunctionError(RootModel[Union[RpcCallFunctionErrorUnknownBlock, RpcCallFunctionErrorInvalidAccount, RpcCallFunctionErrorUnknownAccount, RpcCallFunctionErrorNoContractCode, RpcCallFunctionErrorContractExecutionError, RpcCallFunctionErrorInternalError]]):
    pass

