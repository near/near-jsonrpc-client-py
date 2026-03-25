from near_jsonrpc_models.rpc_receipt_to_tx_request import RpcReceiptToTxRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalReceiptToTx(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_receipt_to_tx']
    params: RpcReceiptToTxRequest
