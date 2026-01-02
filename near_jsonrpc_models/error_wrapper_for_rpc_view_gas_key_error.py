from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from near_jsonrpc_models.rpc_view_gas_key_error import RpcViewGasKeyError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcViewGasKeyErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcViewGasKeyErrorHandlerError(BaseModel):
    cause: RpcViewGasKeyError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcViewGasKeyErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcViewGasKeyError(RootModel[Union[ErrorWrapperForRpcViewGasKeyErrorRequestValidationError, ErrorWrapperForRpcViewGasKeyErrorHandlerError, ErrorWrapperForRpcViewGasKeyErrorInternalError]]):
    pass

