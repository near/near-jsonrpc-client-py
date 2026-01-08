from near_jsonrpc_models.error_wrapper_for_rpc_status_error import ErrorWrapperForRpcStatusError
from near_jsonrpc_models.rpc_status_response import RpcStatusResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcStatusResponseAndRpcStatusErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcStatusResponse

class JsonRpcResponseForRpcStatusResponseAndRpcStatusErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcStatusError

class JsonRpcResponseForRpcStatusResponseAndRpcStatusError(RootModel[Union[JsonRpcResponseForRpcStatusResponseAndRpcStatusErrorResult, JsonRpcResponseForRpcStatusResponseAndRpcStatusErrorError]]):
    pass

