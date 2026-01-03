from near_jsonrpc_models.rpc_health_request import RpcHealthRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForHealth(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['health']
    params: RpcHealthRequest
