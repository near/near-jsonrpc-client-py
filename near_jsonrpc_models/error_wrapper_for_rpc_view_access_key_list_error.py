from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from near_jsonrpc_models.rpc_view_access_key_list_error import RpcViewAccessKeyListError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcViewAccessKeyListErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcViewAccessKeyListErrorHandlerError(BaseModel):
    cause: RpcViewAccessKeyListError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcViewAccessKeyListErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcViewAccessKeyListError(RootModel[Union[ErrorWrapperForRpcViewAccessKeyListErrorRequestValidationError, ErrorWrapperForRpcViewAccessKeyListErrorHandlerError, ErrorWrapperForRpcViewAccessKeyListErrorInternalError]]):
    pass

