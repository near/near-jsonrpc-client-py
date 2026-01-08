from near_jsonrpc_models.rpc_view_code_request import RpcViewCodeRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalViewCode(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_view_code']
    params: RpcViewCodeRequest
