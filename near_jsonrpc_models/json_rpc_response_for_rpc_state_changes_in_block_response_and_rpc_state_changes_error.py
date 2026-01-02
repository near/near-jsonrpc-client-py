from near_jsonrpc_models.error_wrapper_for_rpc_state_changes_error import ErrorWrapperForRpcStateChangesError
from near_jsonrpc_models.rpc_state_changes_in_block_response import RpcStateChangesInBlockResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcStateChangesInBlockResponseAndRpcStateChangesErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcStateChangesInBlockResponse

class JsonRpcResponseForRpcStateChangesInBlockResponseAndRpcStateChangesErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcStateChangesError

class JsonRpcResponseForRpcStateChangesInBlockResponseAndRpcStateChangesError(RootModel[Union[JsonRpcResponseForRpcStateChangesInBlockResponseAndRpcStateChangesErrorResult, JsonRpcResponseForRpcStateChangesInBlockResponseAndRpcStateChangesErrorError]]):
    pass

