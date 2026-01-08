from near_jsonrpc_models.error_wrapper_for_rpc_view_state_error import ErrorWrapperForRpcViewStateError
from near_jsonrpc_models.rpc_view_state_response import RpcViewStateResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcViewStateResponseAndRpcViewStateErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcViewStateResponse

class JsonRpcResponseForRpcViewStateResponseAndRpcViewStateErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcViewStateError

class JsonRpcResponseForRpcViewStateResponseAndRpcViewStateError(RootModel[Union[JsonRpcResponseForRpcViewStateResponseAndRpcViewStateErrorResult, JsonRpcResponseForRpcViewStateResponseAndRpcViewStateErrorError]]):
    pass

