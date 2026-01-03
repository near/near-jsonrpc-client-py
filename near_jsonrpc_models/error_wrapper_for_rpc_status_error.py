from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from near_jsonrpc_models.rpc_status_error import RpcStatusError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcStatusErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcStatusErrorHandlerError(BaseModel):
    cause: RpcStatusError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcStatusErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcStatusError(RootModel[Union[ErrorWrapperForRpcStatusErrorRequestValidationError, ErrorWrapperForRpcStatusErrorHandlerError, ErrorWrapperForRpcStatusErrorInternalError]]):
    pass

