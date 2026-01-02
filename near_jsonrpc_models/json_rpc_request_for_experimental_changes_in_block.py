from near_jsonrpc_models.rpc_state_changes_in_block_request import RpcStateChangesInBlockRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalChangesInBlock(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_changes_in_block']
    params: RpcStateChangesInBlockRequest
