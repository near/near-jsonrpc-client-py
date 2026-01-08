from near_jsonrpc_models.error_wrapper_for_rpc_query_error import ErrorWrapperForRpcQueryError
from near_jsonrpc_models.rpc_query_response import RpcQueryResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcQueryResponseAndRpcQueryErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcQueryResponse

class JsonRpcResponseForRpcQueryResponseAndRpcQueryErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcQueryError

class JsonRpcResponseForRpcQueryResponseAndRpcQueryError(RootModel[Union[JsonRpcResponseForRpcQueryResponseAndRpcQueryErrorResult, JsonRpcResponseForRpcQueryResponseAndRpcQueryErrorError]]):
    pass

