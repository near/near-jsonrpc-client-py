from near_jsonrpc_models.rpc_call_function_request import RpcCallFunctionRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalCallFunction(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_call_function']
    params: RpcCallFunctionRequest
