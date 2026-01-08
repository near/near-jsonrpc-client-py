from near_jsonrpc_models.rpc_view_access_key_list_request import RpcViewAccessKeyListRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalViewAccessKeyList(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_view_access_key_list']
    params: RpcViewAccessKeyListRequest
