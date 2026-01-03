from near_jsonrpc_models.rpc_view_gas_key_request import RpcViewGasKeyRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalViewGasKey(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_view_gas_key']
    params: RpcViewGasKeyRequest
