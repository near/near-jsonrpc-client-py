from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_query_error import RpcQueryError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcQueryErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcQueryErrorHandlerError(BaseModel):
    cause: RpcQueryError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcQueryErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcQueryError(RootModel[Union[ErrorWrapperForRpcQueryErrorRequestValidationError, ErrorWrapperForRpcQueryErrorHandlerError, ErrorWrapperForRpcQueryErrorInternalError]]):
    pass

