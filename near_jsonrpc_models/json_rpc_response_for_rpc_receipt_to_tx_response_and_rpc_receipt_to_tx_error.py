from near_jsonrpc_models.error_wrapper_for_rpc_receipt_to_tx_error import ErrorWrapperForRpcReceiptToTxError
from near_jsonrpc_models.rpc_receipt_to_tx_response import RpcReceiptToTxResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcReceiptToTxResponseAndRpcReceiptToTxErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcReceiptToTxResponse

class JsonRpcResponseForRpcReceiptToTxResponseAndRpcReceiptToTxErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcReceiptToTxError

class JsonRpcResponseForRpcReceiptToTxResponseAndRpcReceiptToTxError(RootModel[Union[JsonRpcResponseForRpcReceiptToTxResponseAndRpcReceiptToTxErrorResult, JsonRpcResponseForRpcReceiptToTxResponseAndRpcReceiptToTxErrorError]]):
    pass

