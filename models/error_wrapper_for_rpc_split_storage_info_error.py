from models.internal_error import InternalError
from models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from models.rpc_split_storage_info_error import RpcSplitStorageInfoError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcSplitStorageInfoErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcSplitStorageInfoErrorHandlerError(BaseModel):
    cause: RpcSplitStorageInfoError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcSplitStorageInfoErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcSplitStorageInfoError(RootModel[Union[ErrorWrapperForRpcSplitStorageInfoErrorRequestValidationError, ErrorWrapperForRpcSplitStorageInfoErrorHandlerError, ErrorWrapperForRpcSplitStorageInfoErrorInternalError]]):
    pass

