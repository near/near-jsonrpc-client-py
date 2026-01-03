from near_jsonrpc_models.genesis_config_request import GenesisConfigRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalGenesisConfig(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_genesis_config']
    params: GenesisConfigRequest
