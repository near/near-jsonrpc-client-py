from near_jsonrpc_models.error_wrapper_for_rpc_client_config_error import ErrorWrapperForRpcClientConfigError
from near_jsonrpc_models.rpc_client_config_response import RpcClientConfigResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcClientConfigResponseAndRpcClientConfigErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcClientConfigResponse

class JsonRpcResponseForRpcClientConfigResponseAndRpcClientConfigErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcClientConfigError

class JsonRpcResponseForRpcClientConfigResponseAndRpcClientConfigError(RootModel[Union[JsonRpcResponseForRpcClientConfigResponseAndRpcClientConfigErrorResult, JsonRpcResponseForRpcClientConfigResponseAndRpcClientConfigErrorError]]):
    pass

