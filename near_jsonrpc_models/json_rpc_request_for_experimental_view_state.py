from near_jsonrpc_models.rpc_view_state_request import RpcViewStateRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalViewState(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_view_state']
    params: RpcViewStateRequest
