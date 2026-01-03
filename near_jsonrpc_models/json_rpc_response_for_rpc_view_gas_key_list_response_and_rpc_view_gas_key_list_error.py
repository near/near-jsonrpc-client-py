from near_jsonrpc_models.error_wrapper_for_rpc_view_gas_key_list_error import ErrorWrapperForRpcViewGasKeyListError
from near_jsonrpc_models.rpc_view_gas_key_list_response import RpcViewGasKeyListResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcViewGasKeyListResponseAndRpcViewGasKeyListErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcViewGasKeyListResponse

class JsonRpcResponseForRpcViewGasKeyListResponseAndRpcViewGasKeyListErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcViewGasKeyListError

class JsonRpcResponseForRpcViewGasKeyListResponseAndRpcViewGasKeyListError(RootModel[Union[JsonRpcResponseForRpcViewGasKeyListResponseAndRpcViewGasKeyListErrorResult, JsonRpcResponseForRpcViewGasKeyListResponseAndRpcViewGasKeyListErrorError]]):
    pass

