from near_jsonrpc_models.error_wrapper_for_genesis_config_error import ErrorWrapperForGenesisConfigError
from near_jsonrpc_models.genesis_config import GenesisConfig
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForGenesisConfigAndGenesisConfigErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: GenesisConfig

class JsonRpcResponseForGenesisConfigAndGenesisConfigErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForGenesisConfigError

class JsonRpcResponseForGenesisConfigAndGenesisConfigError(RootModel[Union[JsonRpcResponseForGenesisConfigAndGenesisConfigErrorResult, JsonRpcResponseForGenesisConfigAndGenesisConfigErrorError]]):
    pass

