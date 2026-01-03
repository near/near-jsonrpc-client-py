from near_jsonrpc_models.rpc_congestion_level_request import RpcCongestionLevelRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalCongestionLevel(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_congestion_level']
    params: RpcCongestionLevelRequest
