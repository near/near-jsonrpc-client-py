from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_reference import BlockReference
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class RpcViewCodeErrorUnknownBlockInfo(BaseModel):
    block_reference: BlockReference

class RpcViewCodeErrorUnknownBlock(BaseModel):
    info: RpcViewCodeErrorUnknownBlockInfo
    name: Literal['UNKNOWN_BLOCK']

class RpcViewCodeErrorInvalidAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewCodeErrorInvalidAccount(BaseModel):
    info: RpcViewCodeErrorInvalidAccountInfo
    name: Literal['INVALID_ACCOUNT']

class RpcViewCodeErrorUnknownAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewCodeErrorUnknownAccount(BaseModel):
    info: RpcViewCodeErrorUnknownAccountInfo
    name: Literal['UNKNOWN_ACCOUNT']

class RpcViewCodeErrorNoContractCodeInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    contract_account_id: AccountId

class RpcViewCodeErrorNoContractCode(BaseModel):
    info: RpcViewCodeErrorNoContractCodeInfo
    name: Literal['NO_CONTRACT_CODE']

class RpcViewCodeErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcViewCodeErrorInternalError(BaseModel):
    info: RpcViewCodeErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcViewCodeError(RootModel[Union[RpcViewCodeErrorUnknownBlock, RpcViewCodeErrorInvalidAccount, RpcViewCodeErrorUnknownAccount, RpcViewCodeErrorNoContractCode, RpcViewCodeErrorInternalError]]):
    pass

