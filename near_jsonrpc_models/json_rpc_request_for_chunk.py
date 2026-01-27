from near_jsonrpc_models.rpc_chunk_request import RpcChunkRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForChunk(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['chunk']
    params: RpcChunkRequest
