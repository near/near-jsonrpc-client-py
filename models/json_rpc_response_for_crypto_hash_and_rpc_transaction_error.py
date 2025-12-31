from models.crypto_hash import CryptoHash
from models.error_wrapper_for_rpc_transaction_error import ErrorWrapperForRpcTransactionError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForCryptoHashAndRpcTransactionErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: CryptoHash

class JsonRpcResponseForCryptoHashAndRpcTransactionErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcTransactionError

class JsonRpcResponseForCryptoHashAndRpcTransactionError(RootModel[Union[JsonRpcResponseForCryptoHashAndRpcTransactionErrorResult, JsonRpcResponseForCryptoHashAndRpcTransactionErrorError]]):
    pass

