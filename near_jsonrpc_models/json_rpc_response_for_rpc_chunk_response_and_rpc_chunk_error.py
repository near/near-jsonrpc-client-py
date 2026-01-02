from near_jsonrpc_models.error_wrapper_for_rpc_chunk_error import ErrorWrapperForRpcChunkError
from near_jsonrpc_models.rpc_chunk_response import RpcChunkResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcChunkResponseAndRpcChunkErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcChunkResponse

class JsonRpcResponseForRpcChunkResponseAndRpcChunkErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcChunkError

class JsonRpcResponseForRpcChunkResponseAndRpcChunkError(RootModel[Union[JsonRpcResponseForRpcChunkResponseAndRpcChunkErrorResult, JsonRpcResponseForRpcChunkResponseAndRpcChunkErrorError]]):
    pass

