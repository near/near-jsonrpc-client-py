from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_client_config_error import RpcClientConfigError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcClientConfigErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcClientConfigErrorHandlerError(BaseModel):
    cause: RpcClientConfigError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcClientConfigErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcClientConfigError(RootModel[Union[ErrorWrapperForRpcClientConfigErrorRequestValidationError, ErrorWrapperForRpcClientConfigErrorHandlerError, ErrorWrapperForRpcClientConfigErrorInternalError]]):
    pass

