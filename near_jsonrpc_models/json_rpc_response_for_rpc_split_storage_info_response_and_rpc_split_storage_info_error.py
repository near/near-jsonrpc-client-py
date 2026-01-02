from near_jsonrpc_models.error_wrapper_for_rpc_split_storage_info_error import ErrorWrapperForRpcSplitStorageInfoError
from near_jsonrpc_models.rpc_split_storage_info_response import RpcSplitStorageInfoResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcSplitStorageInfoResponseAndRpcSplitStorageInfoErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcSplitStorageInfoResponse

class JsonRpcResponseForRpcSplitStorageInfoResponseAndRpcSplitStorageInfoErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcSplitStorageInfoError

class JsonRpcResponseForRpcSplitStorageInfoResponseAndRpcSplitStorageInfoError(RootModel[Union[JsonRpcResponseForRpcSplitStorageInfoResponseAndRpcSplitStorageInfoErrorResult, JsonRpcResponseForRpcSplitStorageInfoResponseAndRpcSplitStorageInfoErrorError]]):
    pass

