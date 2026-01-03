from near_jsonrpc_models.rpc_view_gas_key_list_request import RpcViewGasKeyListRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalViewGasKeyList(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_view_gas_key_list']
    params: RpcViewGasKeyListRequest
