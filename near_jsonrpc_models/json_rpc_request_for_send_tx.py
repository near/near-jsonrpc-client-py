from near_jsonrpc_models.rpc_send_transaction_request import RpcSendTransactionRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForSendTx(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['send_tx']
    params: RpcSendTransactionRequest
