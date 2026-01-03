from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_reference import BlockReference
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class RpcViewAccessKeyListErrorUnknownBlockInfo(BaseModel):
    block_reference: BlockReference

class RpcViewAccessKeyListErrorUnknownBlock(BaseModel):
    info: RpcViewAccessKeyListErrorUnknownBlockInfo
    name: Literal['UNKNOWN_BLOCK']

class RpcViewAccessKeyListErrorInvalidAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewAccessKeyListErrorInvalidAccount(BaseModel):
    info: RpcViewAccessKeyListErrorInvalidAccountInfo
    name: Literal['INVALID_ACCOUNT']

class RpcViewAccessKeyListErrorUnknownAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewAccessKeyListErrorUnknownAccount(BaseModel):
    info: RpcViewAccessKeyListErrorUnknownAccountInfo
    name: Literal['UNKNOWN_ACCOUNT']

class RpcViewAccessKeyListErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcViewAccessKeyListErrorInternalError(BaseModel):
    info: RpcViewAccessKeyListErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcViewAccessKeyListError(RootModel[Union[RpcViewAccessKeyListErrorUnknownBlock, RpcViewAccessKeyListErrorInvalidAccount, RpcViewAccessKeyListErrorUnknownAccount, RpcViewAccessKeyListErrorInternalError]]):
    pass

