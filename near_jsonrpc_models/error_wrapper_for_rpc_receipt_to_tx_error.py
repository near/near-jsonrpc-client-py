from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_receipt_to_tx_error import RpcReceiptToTxError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcReceiptToTxErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcReceiptToTxErrorHandlerError(BaseModel):
    cause: RpcReceiptToTxError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcReceiptToTxErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcReceiptToTxError(RootModel[Union[ErrorWrapperForRpcReceiptToTxErrorRequestValidationError, ErrorWrapperForRpcReceiptToTxErrorHandlerError, ErrorWrapperForRpcReceiptToTxErrorInternalError]]):
    pass

