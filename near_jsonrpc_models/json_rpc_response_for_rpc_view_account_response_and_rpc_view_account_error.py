from near_jsonrpc_models.error_wrapper_for_rpc_view_account_error import ErrorWrapperForRpcViewAccountError
from near_jsonrpc_models.rpc_view_account_response import RpcViewAccountResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcViewAccountResponseAndRpcViewAccountErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcViewAccountResponse

class JsonRpcResponseForRpcViewAccountResponseAndRpcViewAccountErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcViewAccountError

class JsonRpcResponseForRpcViewAccountResponseAndRpcViewAccountError(RootModel[Union[JsonRpcResponseForRpcViewAccountResponseAndRpcViewAccountErrorResult, JsonRpcResponseForRpcViewAccountResponseAndRpcViewAccountErrorError]]):
    pass

