from near_jsonrpc_models.error_wrapper_for_rpc_receipt_error import ErrorWrapperForRpcReceiptError
from near_jsonrpc_models.rpc_receipt_response import RpcReceiptResponse
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class JsonRpcResponseForRpcReceiptResponseAndRpcReceiptErrorResult(BaseModel):
    id: str
    jsonrpc: str
    result: RpcReceiptResponse

class JsonRpcResponseForRpcReceiptResponseAndRpcReceiptErrorError(BaseModel):
    id: str
    jsonrpc: str
    error: ErrorWrapperForRpcReceiptError

class JsonRpcResponseForRpcReceiptResponseAndRpcReceiptError(RootModel[Union[JsonRpcResponseForRpcReceiptResponseAndRpcReceiptErrorResult, JsonRpcResponseForRpcReceiptResponseAndRpcReceiptErrorError]]):
    pass

