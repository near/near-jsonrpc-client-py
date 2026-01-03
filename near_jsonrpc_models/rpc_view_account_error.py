from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_reference import BlockReference
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class RpcViewAccountErrorUnknownBlockInfo(BaseModel):
    block_reference: BlockReference

class RpcViewAccountErrorUnknownBlock(BaseModel):
    info: RpcViewAccountErrorUnknownBlockInfo
    name: Literal['UNKNOWN_BLOCK']

class RpcViewAccountErrorInvalidAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewAccountErrorInvalidAccount(BaseModel):
    info: RpcViewAccountErrorInvalidAccountInfo
    name: Literal['INVALID_ACCOUNT']

class RpcViewAccountErrorUnknownAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewAccountErrorUnknownAccount(BaseModel):
    info: RpcViewAccountErrorUnknownAccountInfo
    name: Literal['UNKNOWN_ACCOUNT']

class RpcViewAccountErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcViewAccountErrorInternalError(BaseModel):
    info: RpcViewAccountErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcViewAccountError(RootModel[Union[RpcViewAccountErrorUnknownBlock, RpcViewAccountErrorInvalidAccount, RpcViewAccountErrorUnknownAccount, RpcViewAccountErrorInternalError]]):
    pass

