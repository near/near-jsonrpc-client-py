from near_jsonrpc_models.error_wrapper_for_rpc_view_access_key_list_error import ErrorWrapperForRpcViewAccessKeyListError
from near_jsonrpc_models.rpc_view_access_key_list_response import RpcViewAccessKeyListResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcViewAccessKeyListResponseAndRpcViewAccessKeyListErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcViewAccessKeyListResponse

class JsonRpcResponseForRpcViewAccessKeyListResponseAndRpcViewAccessKeyListErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcViewAccessKeyListError

class JsonRpcResponseForRpcViewAccessKeyListResponseAndRpcViewAccessKeyListError(RootModel[Union[JsonRpcResponseForRpcViewAccessKeyListResponseAndRpcViewAccessKeyListErrorResult, JsonRpcResponseForRpcViewAccessKeyListResponseAndRpcViewAccessKeyListErrorError]]):
    pass

