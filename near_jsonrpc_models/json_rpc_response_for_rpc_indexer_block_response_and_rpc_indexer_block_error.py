from near_jsonrpc_models.error_wrapper_for_rpc_indexer_block_error import ErrorWrapperForRpcIndexerBlockError
from near_jsonrpc_models.rpc_indexer_block_response import RpcIndexerBlockResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcIndexerBlockResponseAndRpcIndexerBlockErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcIndexerBlockResponse

class JsonRpcResponseForRpcIndexerBlockResponseAndRpcIndexerBlockErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcIndexerBlockError

class JsonRpcResponseForRpcIndexerBlockResponseAndRpcIndexerBlockError(RootModel[Union[JsonRpcResponseForRpcIndexerBlockResponseAndRpcIndexerBlockErrorResult, JsonRpcResponseForRpcIndexerBlockResponseAndRpcIndexerBlockErrorError]]):
    pass

