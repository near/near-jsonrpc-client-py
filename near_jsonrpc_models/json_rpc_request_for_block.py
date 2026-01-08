from near_jsonrpc_models.rpc_block_request import RpcBlockRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForBlock(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['block']
    params: RpcBlockRequest
