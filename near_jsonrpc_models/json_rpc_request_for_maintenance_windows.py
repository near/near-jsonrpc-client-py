from near_jsonrpc_models.rpc_maintenance_windows_request import RpcMaintenanceWindowsRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForMaintenanceWindows(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['maintenance_windows']
    params: RpcMaintenanceWindowsRequest
