from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class RpcReceiptToTxErrorUnknownReceiptInfo(BaseModel):
    receipt_id: CryptoHash

class RpcReceiptToTxErrorUnknownReceipt(BaseModel):
    info: RpcReceiptToTxErrorUnknownReceiptInfo
    name: Literal['UNKNOWN_RECEIPT']

class RpcReceiptToTxErrorDepthExceededInfo(BaseModel):
    limit: conint(ge=0, le=4294967295)
    receipt_id: CryptoHash

class RpcReceiptToTxErrorDepthExceeded(BaseModel):
    info: RpcReceiptToTxErrorDepthExceededInfo
    name: Literal['DEPTH_EXCEEDED']

class RpcReceiptToTxErrorUnsupportedInfo(BaseModel):
    error_message: str

class RpcReceiptToTxErrorUnsupported(BaseModel):
    info: RpcReceiptToTxErrorUnsupportedInfo
    name: Literal['UNSUPPORTED']

class RpcReceiptToTxErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcReceiptToTxErrorInternalError(BaseModel):
    info: RpcReceiptToTxErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcReceiptToTxError(RootModel[Union[RpcReceiptToTxErrorUnknownReceipt, RpcReceiptToTxErrorDepthExceeded, RpcReceiptToTxErrorUnsupported, RpcReceiptToTxErrorInternalError]]):
    pass

