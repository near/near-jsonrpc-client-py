from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_chunk_error import RpcChunkError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcChunkErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcChunkErrorHandlerError(BaseModel):
    cause: RpcChunkError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcChunkErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcChunkError(RootModel[Union[ErrorWrapperForRpcChunkErrorRequestValidationError, ErrorWrapperForRpcChunkErrorHandlerError, ErrorWrapperForRpcChunkErrorInternalError]]):
    pass

