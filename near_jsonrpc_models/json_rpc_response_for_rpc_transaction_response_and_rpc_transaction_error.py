from near_jsonrpc_models.error_wrapper_for_rpc_transaction_error import ErrorWrapperForRpcTransactionError
from near_jsonrpc_models.rpc_transaction_response import RpcTransactionResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcTransactionResponseAndRpcTransactionErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcTransactionResponse

class JsonRpcResponseForRpcTransactionResponseAndRpcTransactionErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcTransactionError

class JsonRpcResponseForRpcTransactionResponseAndRpcTransactionError(RootModel[Union[JsonRpcResponseForRpcTransactionResponseAndRpcTransactionErrorResult, JsonRpcResponseForRpcTransactionResponseAndRpcTransactionErrorError]]):
    pass

