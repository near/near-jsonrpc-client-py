from near_jsonrpc_models.rpc_light_client_next_block_request import RpcLightClientNextBlockRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForNextLightClientBlock(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['next_light_client_block']
    params: RpcLightClientNextBlockRequest
