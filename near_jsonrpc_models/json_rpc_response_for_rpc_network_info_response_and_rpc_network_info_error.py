from near_jsonrpc_models.error_wrapper_for_rpc_network_info_error import ErrorWrapperForRpcNetworkInfoError
from near_jsonrpc_models.rpc_network_info_response import RpcNetworkInfoResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcNetworkInfoResponseAndRpcNetworkInfoErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcNetworkInfoResponse

class JsonRpcResponseForRpcNetworkInfoResponseAndRpcNetworkInfoErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcNetworkInfoError

class JsonRpcResponseForRpcNetworkInfoResponseAndRpcNetworkInfoError(RootModel[Union[JsonRpcResponseForRpcNetworkInfoResponseAndRpcNetworkInfoErrorResult, JsonRpcResponseForRpcNetworkInfoResponseAndRpcNetworkInfoErrorError]]):
    pass

