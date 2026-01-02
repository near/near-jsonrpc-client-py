from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from near_jsonrpc_models.rpc_view_code_error import RpcViewCodeError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcViewCodeErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcViewCodeErrorHandlerError(BaseModel):
    cause: RpcViewCodeError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcViewCodeErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcViewCodeError(RootModel[Union[ErrorWrapperForRpcViewCodeErrorRequestValidationError, ErrorWrapperForRpcViewCodeErrorHandlerError, ErrorWrapperForRpcViewCodeErrorInternalError]]):
    pass

