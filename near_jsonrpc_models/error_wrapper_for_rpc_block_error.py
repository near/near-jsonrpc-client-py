from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_block_error import RpcBlockError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcBlockErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcBlockErrorHandlerError(BaseModel):
    cause: RpcBlockError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcBlockErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcBlockError(RootModel[Union[ErrorWrapperForRpcBlockErrorRequestValidationError, ErrorWrapperForRpcBlockErrorHandlerError, ErrorWrapperForRpcBlockErrorInternalError]]):
    pass

