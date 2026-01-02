from near_jsonrpc_models.rpc_receipt_request import RpcReceiptRequest
from pydantic import BaseModel
from typing import Literal


class JsonRpcRequestForExperimentalReceipt(BaseModel):
    id: str
    jsonrpc: str
    method: Literal['EXPERIMENTAL_receipt']
    params: RpcReceiptRequest
