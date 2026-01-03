from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from near_jsonrpc_models.rpc_view_state_error import RpcViewStateError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcViewStateErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcViewStateErrorHandlerError(BaseModel):
    cause: RpcViewStateError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcViewStateErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcViewStateError(RootModel[Union[ErrorWrapperForRpcViewStateErrorRequestValidationError, ErrorWrapperForRpcViewStateErrorHandlerError, ErrorWrapperForRpcViewStateErrorInternalError]]):
    pass

