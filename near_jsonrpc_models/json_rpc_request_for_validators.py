from near_jsonrpc_models.rpc_validator_request import RpcValidatorRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForValidators(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['validators']
    params: RpcValidatorRequest
