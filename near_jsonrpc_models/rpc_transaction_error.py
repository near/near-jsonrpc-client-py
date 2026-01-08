from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import RootModel
from typing import Any
from typing import Dict
from typing import Literal
from typing import Union


class RpcTransactionErrorInvalidTransaction(BaseModel):
    info: Dict[str, Any]
    name: Literal['INVALID_TRANSACTION']

class RpcTransactionErrorDoesNotTrackShard(BaseModel):
    name: Literal['DOES_NOT_TRACK_SHARD']

class RpcTransactionErrorRequestRoutedInfo(BaseModel):
    transaction_hash: CryptoHash

class RpcTransactionErrorRequestRouted(BaseModel):
    info: RpcTransactionErrorRequestRoutedInfo
    name: Literal['REQUEST_ROUTED']

class RpcTransactionErrorUnknownTransactionInfo(BaseModel):
    requested_transaction_hash: CryptoHash

class RpcTransactionErrorUnknownTransaction(BaseModel):
    info: RpcTransactionErrorUnknownTransactionInfo
    name: Literal['UNKNOWN_TRANSACTION']

class RpcTransactionErrorInternalErrorInfo(BaseModel):
    debug_info: str

class RpcTransactionErrorInternalError(BaseModel):
    info: RpcTransactionErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcTransactionErrorTimeoutError(BaseModel):
    name: Literal['TIMEOUT_ERROR']

class RpcTransactionError(RootModel[Union[RpcTransactionErrorInvalidTransaction, RpcTransactionErrorDoesNotTrackShard, RpcTransactionErrorRequestRouted, RpcTransactionErrorUnknownTransaction, RpcTransactionErrorInternalError, RpcTransactionErrorTimeoutError]]):
    pass

