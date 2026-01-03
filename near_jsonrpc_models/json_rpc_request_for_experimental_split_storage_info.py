from near_jsonrpc_models.rpc_split_storage_info_request import RpcSplitStorageInfoRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalSplitStorageInfo(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_split_storage_info']
    params: RpcSplitStorageInfoRequest
