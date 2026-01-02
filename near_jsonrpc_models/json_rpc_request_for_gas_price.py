from near_jsonrpc_models.rpc_gas_price_request import RpcGasPriceRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForGasPrice(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['gas_price']
    params: RpcGasPriceRequest
