from near_jsonrpc_models.rpc_view_access_key_request import RpcViewAccessKeyRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalViewAccessKey(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_view_access_key']
    params: RpcViewAccessKeyRequest
