from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_call_function_error import RpcCallFunctionError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcCallFunctionErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcCallFunctionErrorHandlerError(BaseModel):
    cause: RpcCallFunctionError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcCallFunctionErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcCallFunctionError(RootModel[Union[ErrorWrapperForRpcCallFunctionErrorRequestValidationError, ErrorWrapperForRpcCallFunctionErrorHandlerError, ErrorWrapperForRpcCallFunctionErrorInternalError]]):
    pass

