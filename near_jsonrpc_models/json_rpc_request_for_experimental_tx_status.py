from near_jsonrpc_models.rpc_transaction_status_request import RpcTransactionStatusRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalTxStatus(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_tx_status']
    params: RpcTransactionStatusRequest
