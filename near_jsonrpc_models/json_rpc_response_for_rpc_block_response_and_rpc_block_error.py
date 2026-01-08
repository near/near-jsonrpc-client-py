from near_jsonrpc_models.error_wrapper_for_rpc_block_error import ErrorWrapperForRpcBlockError
from near_jsonrpc_models.rpc_block_response import RpcBlockResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcBlockResponseAndRpcBlockErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcBlockResponse

class JsonRpcResponseForRpcBlockResponseAndRpcBlockErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcBlockError

class JsonRpcResponseForRpcBlockResponseAndRpcBlockError(RootModel[Union[JsonRpcResponseForRpcBlockResponseAndRpcBlockErrorResult, JsonRpcResponseForRpcBlockResponseAndRpcBlockErrorError]]):
    pass

