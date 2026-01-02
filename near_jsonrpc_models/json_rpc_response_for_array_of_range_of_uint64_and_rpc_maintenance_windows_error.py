from near_jsonrpc_models.error_wrapper_for_rpc_maintenance_windows_error import ErrorWrapperForRpcMaintenanceWindowsError
from near_jsonrpc_models.range_of_uint64 import RangeOfUint64
from pydantic import BaseModel
from pydantic import RootModel
from typing import List
from typing import Union


class JsonRpcResponseForArrayOfRangeOfUint64AndRpcMaintenanceWindowsErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: List[RangeOfUint64]

class JsonRpcResponseForArrayOfRangeOfUint64AndRpcMaintenanceWindowsErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcMaintenanceWindowsError

class JsonRpcResponseForArrayOfRangeOfUint64AndRpcMaintenanceWindowsError(RootModel[Union[JsonRpcResponseForArrayOfRangeOfUint64AndRpcMaintenanceWindowsErrorResult, JsonRpcResponseForArrayOfRangeOfUint64AndRpcMaintenanceWindowsErrorError]]):
    pass

