from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_indexer_block_error import RpcIndexerBlockError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcIndexerBlockErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcIndexerBlockErrorHandlerError(BaseModel):
    cause: RpcIndexerBlockError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcIndexerBlockErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcIndexerBlockError(RootModel[Union[ErrorWrapperForRpcIndexerBlockErrorRequestValidationError, ErrorWrapperForRpcIndexerBlockErrorHandlerError, ErrorWrapperForRpcIndexerBlockErrorInternalError]]):
    pass

