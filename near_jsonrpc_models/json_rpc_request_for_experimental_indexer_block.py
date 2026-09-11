from near_jsonrpc_models.rpc_indexer_block_request import RpcIndexerBlockRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalIndexerBlock(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_indexer_block']
    params: RpcIndexerBlockRequest
