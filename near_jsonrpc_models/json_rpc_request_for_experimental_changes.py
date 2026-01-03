from near_jsonrpc_models.rpc_state_changes_in_block_by_type_request import RpcStateChangesInBlockByTypeRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalChanges(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_changes']
    params: RpcStateChangesInBlockByTypeRequest
