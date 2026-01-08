from near_jsonrpc_models.rpc_network_info_request import RpcNetworkInfoRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForNetworkInfo(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['network_info']
    params: RpcNetworkInfoRequest
