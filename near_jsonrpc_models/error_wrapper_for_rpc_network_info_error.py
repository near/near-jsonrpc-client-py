from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_network_info_error import RpcNetworkInfoError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcNetworkInfoErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcNetworkInfoErrorHandlerError(BaseModel):
    cause: RpcNetworkInfoError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcNetworkInfoErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcNetworkInfoError(RootModel[Union[ErrorWrapperForRpcNetworkInfoErrorRequestValidationError, ErrorWrapperForRpcNetworkInfoErrorHandlerError, ErrorWrapperForRpcNetworkInfoErrorInternalError]]):
    pass

