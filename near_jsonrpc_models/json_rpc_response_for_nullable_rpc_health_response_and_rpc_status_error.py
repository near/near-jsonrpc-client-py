from near_jsonrpc_models.error_wrapper_for_rpc_status_error import ErrorWrapperForRpcStatusError
from near_jsonrpc_models.rpc_health_response import RpcHealthResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForNullableRpcHealthResponseAndRpcStatusErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcHealthResponse | None

class JsonRpcResponseForNullableRpcHealthResponseAndRpcStatusErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcStatusError

class JsonRpcResponseForNullableRpcHealthResponseAndRpcStatusError(RootModel[Union[JsonRpcResponseForNullableRpcHealthResponseAndRpcStatusErrorResult, JsonRpcResponseForNullableRpcHealthResponseAndRpcStatusErrorError]]):
    pass

