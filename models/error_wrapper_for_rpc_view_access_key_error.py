from models.internal_error import InternalError
from models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from models.rpc_view_access_key_error import RpcViewAccessKeyError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcViewAccessKeyErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcViewAccessKeyErrorHandlerError(BaseModel):
    cause: RpcViewAccessKeyError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcViewAccessKeyErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcViewAccessKeyError(RootModel[Union[ErrorWrapperForRpcViewAccessKeyErrorRequestValidationError, ErrorWrapperForRpcViewAccessKeyErrorHandlerError, ErrorWrapperForRpcViewAccessKeyErrorInternalError]]):
    pass

