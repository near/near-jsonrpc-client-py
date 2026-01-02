from near_jsonrpc_models.rpc_protocol_config_request import RpcProtocolConfigRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalProtocolConfig(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_protocol_config']
    params: RpcProtocolConfigRequest
