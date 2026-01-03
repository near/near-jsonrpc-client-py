from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class RpcReceiptErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcReceiptErrorInternalError(BaseModel):
    info: RpcReceiptErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcReceiptErrorUnknownReceiptInfo(BaseModel):
    receipt_id: CryptoHash

class RpcReceiptErrorUnknownReceipt(BaseModel):
    info: RpcReceiptErrorUnknownReceiptInfo
    name: Literal['UNKNOWN_RECEIPT']

class RpcReceiptError(RootModel[Union[RpcReceiptErrorInternalError, RpcReceiptErrorUnknownReceipt]]):
    pass

