from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_reference import BlockReference
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class RpcViewAccessKeyErrorUnknownBlockInfo(BaseModel):
    block_reference: BlockReference

class RpcViewAccessKeyErrorUnknownBlock(BaseModel):
    info: RpcViewAccessKeyErrorUnknownBlockInfo
    name: Literal['UNKNOWN_BLOCK']

class RpcViewAccessKeyErrorInvalidAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewAccessKeyErrorInvalidAccount(BaseModel):
    info: RpcViewAccessKeyErrorInvalidAccountInfo
    name: Literal['INVALID_ACCOUNT']

class RpcViewAccessKeyErrorUnknownAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewAccessKeyErrorUnknownAccount(BaseModel):
    info: RpcViewAccessKeyErrorUnknownAccountInfo
    name: Literal['UNKNOWN_ACCOUNT']

class RpcViewAccessKeyErrorUnknownAccessKeyInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    public_key: PublicKey

class RpcViewAccessKeyErrorUnknownAccessKey(BaseModel):
    info: RpcViewAccessKeyErrorUnknownAccessKeyInfo
    name: Literal['UNKNOWN_ACCESS_KEY']

class RpcViewAccessKeyErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcViewAccessKeyErrorInternalError(BaseModel):
    info: RpcViewAccessKeyErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcViewAccessKeyError(RootModel[Union[RpcViewAccessKeyErrorUnknownBlock, RpcViewAccessKeyErrorInvalidAccount, RpcViewAccessKeyErrorUnknownAccount, RpcViewAccessKeyErrorUnknownAccessKey, RpcViewAccessKeyErrorInternalError]]):
    pass

