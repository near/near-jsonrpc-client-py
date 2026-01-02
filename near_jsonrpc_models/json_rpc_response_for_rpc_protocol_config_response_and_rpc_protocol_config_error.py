from near_jsonrpc_models.error_wrapper_for_rpc_protocol_config_error import ErrorWrapperForRpcProtocolConfigError
from near_jsonrpc_models.rpc_protocol_config_response import RpcProtocolConfigResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcProtocolConfigResponseAndRpcProtocolConfigErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcProtocolConfigResponse

class JsonRpcResponseForRpcProtocolConfigResponseAndRpcProtocolConfigErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcProtocolConfigError

class JsonRpcResponseForRpcProtocolConfigResponseAndRpcProtocolConfigError(RootModel[Union[JsonRpcResponseForRpcProtocolConfigResponseAndRpcProtocolConfigErrorResult, JsonRpcResponseForRpcProtocolConfigResponseAndRpcProtocolConfigErrorError]]):
    pass

