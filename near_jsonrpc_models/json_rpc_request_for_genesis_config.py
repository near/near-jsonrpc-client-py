from near_jsonrpc_models.genesis_config_request import GenesisConfigRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForGenesisConfig(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['genesis_config']
    params: GenesisConfigRequest
