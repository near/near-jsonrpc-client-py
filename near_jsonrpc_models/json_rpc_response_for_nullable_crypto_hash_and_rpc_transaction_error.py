from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.error_wrapper_for_rpc_transaction_error import ErrorWrapperForRpcTransactionError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForNullableCryptoHashAndRpcTransactionErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: CryptoHash | None

class JsonRpcResponseForNullableCryptoHashAndRpcTransactionErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcTransactionError

class JsonRpcResponseForNullableCryptoHashAndRpcTransactionError(RootModel[Union[JsonRpcResponseForNullableCryptoHashAndRpcTransactionErrorResult, JsonRpcResponseForNullableCryptoHashAndRpcTransactionErrorError]]):
    pass

