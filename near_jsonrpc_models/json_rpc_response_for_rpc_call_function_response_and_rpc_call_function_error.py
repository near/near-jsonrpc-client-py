from near_jsonrpc_models.error_wrapper_for_rpc_call_function_error import ErrorWrapperForRpcCallFunctionError
from near_jsonrpc_models.rpc_call_function_response import RpcCallFunctionResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcCallFunctionResponseAndRpcCallFunctionErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcCallFunctionResponse

class JsonRpcResponseForRpcCallFunctionResponseAndRpcCallFunctionErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcCallFunctionError

class JsonRpcResponseForRpcCallFunctionResponseAndRpcCallFunctionError(RootModel[Union[JsonRpcResponseForRpcCallFunctionResponseAndRpcCallFunctionErrorResult, JsonRpcResponseForRpcCallFunctionResponseAndRpcCallFunctionErrorError]]):
    pass

