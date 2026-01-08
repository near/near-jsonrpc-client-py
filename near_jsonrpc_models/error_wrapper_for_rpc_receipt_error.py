from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_receipt_error import RpcReceiptError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcReceiptErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcReceiptErrorHandlerError(BaseModel):
    cause: RpcReceiptError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcReceiptErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcReceiptError(RootModel[Union[ErrorWrapperForRpcReceiptErrorRequestValidationError, ErrorWrapperForRpcReceiptErrorHandlerError, ErrorWrapperForRpcReceiptErrorInternalError]]):
    pass

