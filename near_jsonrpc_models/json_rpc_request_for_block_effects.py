from near_jsonrpc_models.rpc_state_changes_in_block_request import RpcStateChangesInBlockRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForBlockEffects(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['block_effects']
    params: RpcStateChangesInBlockRequest
