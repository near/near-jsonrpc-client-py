from near_jsonrpc_models.rpc_transaction_status_request import RpcTransactionStatusRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForTx(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['tx']
    params: RpcTransactionStatusRequest
