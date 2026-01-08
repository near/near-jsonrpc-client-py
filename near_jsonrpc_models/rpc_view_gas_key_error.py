from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_reference import BlockReference
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class RpcViewGasKeyErrorUnknownBlockInfo(BaseModel):
    block_reference: BlockReference

class RpcViewGasKeyErrorUnknownBlock(BaseModel):
    info: RpcViewGasKeyErrorUnknownBlockInfo
    name: Literal['UNKNOWN_BLOCK']

class RpcViewGasKeyErrorInvalidAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewGasKeyErrorInvalidAccount(BaseModel):
    info: RpcViewGasKeyErrorInvalidAccountInfo
    name: Literal['INVALID_ACCOUNT']

class RpcViewGasKeyErrorUnknownAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewGasKeyErrorUnknownAccount(BaseModel):
    info: RpcViewGasKeyErrorUnknownAccountInfo
    name: Literal['UNKNOWN_ACCOUNT']

class RpcViewGasKeyErrorUnknownGasKeyInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    public_key: PublicKey

class RpcViewGasKeyErrorUnknownGasKey(BaseModel):
    info: RpcViewGasKeyErrorUnknownGasKeyInfo
    name: Literal['UNKNOWN_GAS_KEY']

class RpcViewGasKeyErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcViewGasKeyErrorInternalError(BaseModel):
    info: RpcViewGasKeyErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcViewGasKeyError(RootModel[Union[RpcViewGasKeyErrorUnknownBlock, RpcViewGasKeyErrorInvalidAccount, RpcViewGasKeyErrorUnknownAccount, RpcViewGasKeyErrorUnknownGasKey, RpcViewGasKeyErrorInternalError]]):
    pass

