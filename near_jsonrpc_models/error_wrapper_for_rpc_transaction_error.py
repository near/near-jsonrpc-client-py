from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from near_jsonrpc_models.rpc_transaction_error import RpcTransactionError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcTransactionErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcTransactionErrorHandlerError(BaseModel):
    cause: RpcTransactionError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcTransactionErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcTransactionError(RootModel[Union[ErrorWrapperForRpcTransactionErrorRequestValidationError, ErrorWrapperForRpcTransactionErrorHandlerError, ErrorWrapperForRpcTransactionErrorInternalError]]):
    pass

