from near_jsonrpc_models.rpc_validators_ordered_request import RpcValidatorsOrderedRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalValidatorsOrdered(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_validators_ordered']
    params: RpcValidatorsOrderedRequest
