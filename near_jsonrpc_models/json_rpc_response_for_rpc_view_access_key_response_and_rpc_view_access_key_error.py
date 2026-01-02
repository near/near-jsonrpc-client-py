from near_jsonrpc_models.error_wrapper_for_rpc_view_access_key_error import ErrorWrapperForRpcViewAccessKeyError
from near_jsonrpc_models.rpc_view_access_key_response import RpcViewAccessKeyResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcViewAccessKeyResponseAndRpcViewAccessKeyErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcViewAccessKeyResponse

class JsonRpcResponseForRpcViewAccessKeyResponseAndRpcViewAccessKeyErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcViewAccessKeyError

class JsonRpcResponseForRpcViewAccessKeyResponseAndRpcViewAccessKeyError(RootModel[Union[JsonRpcResponseForRpcViewAccessKeyResponseAndRpcViewAccessKeyErrorResult, JsonRpcResponseForRpcViewAccessKeyResponseAndRpcViewAccessKeyErrorError]]):
    pass

