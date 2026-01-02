from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from near_jsonrpc_models.rpc_validator_error import RpcValidatorError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcValidatorErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcValidatorErrorHandlerError(BaseModel):
    cause: RpcValidatorError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcValidatorErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcValidatorError(RootModel[Union[ErrorWrapperForRpcValidatorErrorRequestValidationError, ErrorWrapperForRpcValidatorErrorHandlerError, ErrorWrapperForRpcValidatorErrorInternalError]]):
    pass

