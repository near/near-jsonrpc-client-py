from near_jsonrpc_models.error_wrapper_for_rpc_gas_price_error import ErrorWrapperForRpcGasPriceError
from near_jsonrpc_models.rpc_gas_price_response import RpcGasPriceResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcGasPriceResponseAndRpcGasPriceErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcGasPriceResponse

class JsonRpcResponseForRpcGasPriceResponseAndRpcGasPriceErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcGasPriceError

class JsonRpcResponseForRpcGasPriceResponseAndRpcGasPriceError(RootModel[Union[JsonRpcResponseForRpcGasPriceResponseAndRpcGasPriceErrorResult, JsonRpcResponseForRpcGasPriceResponseAndRpcGasPriceErrorError]]):
    pass

