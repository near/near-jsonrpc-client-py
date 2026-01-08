from near_jsonrpc_models.error_wrapper_for_rpc_chunk_error import ErrorWrapperForRpcChunkError
from near_jsonrpc_models.rpc_congestion_level_response import RpcCongestionLevelResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcCongestionLevelResponseAndRpcChunkErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcCongestionLevelResponse

class JsonRpcResponseForRpcCongestionLevelResponseAndRpcChunkErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcChunkError

class JsonRpcResponseForRpcCongestionLevelResponseAndRpcChunkError(RootModel[Union[JsonRpcResponseForRpcCongestionLevelResponseAndRpcChunkErrorResult, JsonRpcResponseForRpcCongestionLevelResponseAndRpcChunkErrorError]]):
    pass

