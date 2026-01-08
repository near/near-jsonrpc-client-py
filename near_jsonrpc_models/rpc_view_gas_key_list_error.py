from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_reference import BlockReference
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class RpcViewGasKeyListErrorUnknownBlockInfo(BaseModel):
    block_reference: BlockReference

class RpcViewGasKeyListErrorUnknownBlock(BaseModel):
    info: RpcViewGasKeyListErrorUnknownBlockInfo
    name: Literal['UNKNOWN_BLOCK']

class RpcViewGasKeyListErrorInvalidAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewGasKeyListErrorInvalidAccount(BaseModel):
    info: RpcViewGasKeyListErrorInvalidAccountInfo
    name: Literal['INVALID_ACCOUNT']

class RpcViewGasKeyListErrorUnknownAccountInfo(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    requested_account_id: AccountId

class RpcViewGasKeyListErrorUnknownAccount(BaseModel):
    info: RpcViewGasKeyListErrorUnknownAccountInfo
    name: Literal['UNKNOWN_ACCOUNT']

class RpcViewGasKeyListErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcViewGasKeyListErrorInternalError(BaseModel):
    info: RpcViewGasKeyListErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcViewGasKeyListError(RootModel[Union[RpcViewGasKeyListErrorUnknownBlock, RpcViewGasKeyListErrorInvalidAccount, RpcViewGasKeyListErrorUnknownAccount, RpcViewGasKeyListErrorInternalError]]):
    pass

