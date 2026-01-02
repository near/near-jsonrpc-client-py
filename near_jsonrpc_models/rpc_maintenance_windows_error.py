from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class RpcMaintenanceWindowsErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcMaintenanceWindowsErrorInternalError(BaseModel):
    info: RpcMaintenanceWindowsErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcMaintenanceWindowsError(RootModel[Union[RpcMaintenanceWindowsErrorInternalError]]):
    pass

