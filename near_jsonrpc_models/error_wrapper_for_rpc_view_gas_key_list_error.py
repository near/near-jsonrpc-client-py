from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from near_jsonrpc_models.rpc_view_gas_key_list_error import RpcViewGasKeyListError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcViewGasKeyListErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcViewGasKeyListErrorHandlerError(BaseModel):
    cause: RpcViewGasKeyListError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcViewGasKeyListErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcViewGasKeyListError(RootModel[Union[ErrorWrapperForRpcViewGasKeyListErrorRequestValidationError, ErrorWrapperForRpcViewGasKeyListErrorHandlerError, ErrorWrapperForRpcViewGasKeyListErrorInternalError]]):
    pass

