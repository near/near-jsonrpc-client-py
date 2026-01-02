from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_protocol_config_error import RpcProtocolConfigError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcProtocolConfigErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcProtocolConfigErrorHandlerError(BaseModel):
    cause: RpcProtocolConfigError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcProtocolConfigErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcProtocolConfigError(RootModel[Union[ErrorWrapperForRpcProtocolConfigErrorRequestValidationError, ErrorWrapperForRpcProtocolConfigErrorHandlerError, ErrorWrapperForRpcProtocolConfigErrorInternalError]]):
    pass

