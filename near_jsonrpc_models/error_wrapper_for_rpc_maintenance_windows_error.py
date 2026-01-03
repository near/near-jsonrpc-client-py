from near_jsonrpc_models.internal_error import InternalError
from near_jsonrpc_models.rpc_maintenance_windows_error import RpcMaintenanceWindowsError
from near_jsonrpc_models.rpc_request_validation_error_kind import RpcRequestValidationErrorKind
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ErrorWrapperForRpcMaintenanceWindowsErrorRequestValidationError(BaseModel):
    cause: RpcRequestValidationErrorKind
    name: Literal['REQUEST_VALIDATION_ERROR']

class ErrorWrapperForRpcMaintenanceWindowsErrorHandlerError(BaseModel):
    cause: RpcMaintenanceWindowsError
    name: Literal['HANDLER_ERROR']

class ErrorWrapperForRpcMaintenanceWindowsErrorInternalError(BaseModel):
    cause: InternalError
    name: Literal['INTERNAL_ERROR']

class ErrorWrapperForRpcMaintenanceWindowsError(RootModel[Union[ErrorWrapperForRpcMaintenanceWindowsErrorRequestValidationError, ErrorWrapperForRpcMaintenanceWindowsErrorHandlerError, ErrorWrapperForRpcMaintenanceWindowsErrorInternalError]]):
    pass

