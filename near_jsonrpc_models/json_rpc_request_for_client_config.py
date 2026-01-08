from near_jsonrpc_models.rpc_client_config_request import RpcClientConfigRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForClientConfig(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['client_config']
    params: RpcClientConfigRequest
