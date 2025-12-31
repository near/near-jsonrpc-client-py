from models.internal_error import InternalError
from models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from models.rpc_view_account_error import RpcViewAccountError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcViewAccountErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcViewAccountErrorHandlerError(BaseModel):
    cause: RpcViewAccountError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcViewAccountErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcViewAccountError(RootModel[Union[ErrorWrapperForRpcViewAccountErrorRequestValidationError, ErrorWrapperForRpcViewAccountErrorHandlerError, ErrorWrapperForRpcViewAccountErrorInternalError]]):
    pass

