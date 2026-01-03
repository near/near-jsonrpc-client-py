from near_jsonrpc_models.rpc_status_request import RpcStatusRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForStatus(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['status']
    params: RpcStatusRequest
