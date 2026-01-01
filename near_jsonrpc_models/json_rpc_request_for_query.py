from near_jsonrpc_models.rpc_query_request import RpcQueryRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForQuery(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['query']
    params: RpcQueryRequest
