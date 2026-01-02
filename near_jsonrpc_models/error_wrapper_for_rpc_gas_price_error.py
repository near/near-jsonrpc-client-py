from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_gas_price_error import RpcGasPriceError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcGasPriceErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcGasPriceErrorHandlerError(BaseModel):
    cause: RpcGasPriceError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcGasPriceErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcGasPriceError(RootModel[Union[ErrorWrapperForRpcGasPriceErrorRequestValidationError, ErrorWrapperForRpcGasPriceErrorHandlerError, ErrorWrapperForRpcGasPriceErrorInternalError]]):
    pass

