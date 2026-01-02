from near_jsonrpc_models.error_wrapper_for_rpc_view_code_error import ErrorWrapperForRpcViewCodeError
from near_jsonrpc_models.rpc_view_code_response import RpcViewCodeResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcViewCodeResponseAndRpcViewCodeErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcViewCodeResponse

class JsonRpcResponseForRpcViewCodeResponseAndRpcViewCodeErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcViewCodeError

class JsonRpcResponseForRpcViewCodeResponseAndRpcViewCodeError(RootModel[Union[JsonRpcResponseForRpcViewCodeResponseAndRpcViewCodeErrorResult, JsonRpcResponseForRpcViewCodeResponseAndRpcViewCodeErrorError]]):
    pass

