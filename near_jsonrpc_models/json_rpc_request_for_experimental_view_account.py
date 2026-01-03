from near_jsonrpc_models.rpc_view_account_request import RpcViewAccountRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalViewAccount(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_view_account']
    params: RpcViewAccountRequest
