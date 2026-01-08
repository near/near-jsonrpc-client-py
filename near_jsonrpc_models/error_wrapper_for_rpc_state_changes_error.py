from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from near_jsonrpc_models.rpc_state_changes_error import RpcStateChangesError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcStateChangesErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcStateChangesErrorHandlerError(BaseModel):
    cause: RpcStateChangesError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcStateChangesErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcStateChangesError(RootModel[Union[ErrorWrapperForRpcStateChangesErrorRequestValidationError, ErrorWrapperForRpcStateChangesErrorHandlerError, ErrorWrapperForRpcStateChangesErrorInternalError]]):
    pass

