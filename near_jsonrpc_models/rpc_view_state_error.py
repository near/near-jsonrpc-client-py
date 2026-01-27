from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_reference import BlockReference
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class RpcViewStateErrorUnknownBlockInfo(BaseModel):
    block_reference: BlockReference

class RpcViewStateErrorUnknownBlock(BaseModel):
    info: RpcViewStateErrorUnknownBlockInfo
    name: Literal['UNKNOWN_BLOCK']

class RpcViewStateErrorInvalidAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewStateErrorInvalidAccount(BaseModel):
    info: RpcViewStateErrorInvalidAccountInfo
    name: Literal['INVALID_ACCOUNT']

class RpcViewStateErrorUnknownAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewStateErrorUnknownAccount(BaseModel):
    info: RpcViewStateErrorUnknownAccountInfo
    name: Literal['UNKNOWN_ACCOUNT']

class RpcViewStateErrorTooLargeContractStateInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    contract_account_id: AccountId

class RpcViewStateErrorTooLargeContractState(BaseModel):
    info: RpcViewStateErrorTooLargeContractStateInfo
    name: Literal['TOO_LARGE_CONTRACT_STATE']

class RpcViewStateErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcViewStateErrorInternalError(BaseModel):
    info: RpcViewStateErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcViewStateError(RootModel[Union[RpcViewStateErrorUnknownBlock, RpcViewStateErrorInvalidAccount, RpcViewStateErrorUnknownAccount, RpcViewStateErrorTooLargeContractState, RpcViewStateErrorInternalError]]):
    pass

