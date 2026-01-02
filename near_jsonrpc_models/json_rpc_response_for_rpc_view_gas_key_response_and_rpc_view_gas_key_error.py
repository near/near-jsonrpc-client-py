from near_jsonrpc_models.error_wrapper_for_rpc_view_gas_key_error import ErrorWrapperForRpcViewGasKeyError
from near_jsonrpc_models.rpc_view_gas_key_response import RpcViewGasKeyResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcViewGasKeyResponseAndRpcViewGasKeyErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcViewGasKeyResponse

class JsonRpcResponseForRpcViewGasKeyResponseAndRpcViewGasKeyErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcViewGasKeyError

class JsonRpcResponseForRpcViewGasKeyResponseAndRpcViewGasKeyError(RootModel[Union[JsonRpcResponseForRpcViewGasKeyResponseAndRpcViewGasKeyErrorResult, JsonRpcResponseForRpcViewGasKeyResponseAndRpcViewGasKeyErrorError]]):
    pass

