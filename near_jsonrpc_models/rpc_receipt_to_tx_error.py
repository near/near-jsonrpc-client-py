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

class RpcReceiptToTxErrorOutcomesNotStored(BaseModel):
    name: Literal['OUTCOMES_NOT_STORED']

class RpcReceiptToTxErrorWindowTooLargeInfo(BaseModel):
    maximum: conint(ge=0, le=18446744073709551615)
    requested: conint(ge=0, le=18446744073709551615)

class RpcReceiptToTxErrorWindowTooLarge(BaseModel):
    info: RpcReceiptToTxErrorWindowTooLargeInfo
    name: Literal['WINDOW_TOO_LARGE']

class RpcReceiptToTxErrorMalformedHintInfo(BaseModel):
    error_message: str

class RpcReceiptToTxErrorMalformedHint(BaseModel):
    info: RpcReceiptToTxErrorMalformedHintInfo
    name: Literal['MALFORMED_HINT']

class RpcReceiptToTxErrorBudgetExceededInfo(BaseModel):
    limit: conint(ge=0, le=18446744073709551615)
    scanned: conint(ge=0, le=18446744073709551615)

class RpcReceiptToTxErrorBudgetExceeded(BaseModel):
    info: RpcReceiptToTxErrorBudgetExceededInfo
    name: Literal['BUDGET_EXCEEDED']

class RpcReceiptToTxError(RootModel[Union[RpcReceiptToTxErrorUnknownReceipt, RpcReceiptToTxErrorDepthExceeded, RpcReceiptToTxErrorUnsupported, RpcReceiptToTxErrorInternalError, RpcReceiptToTxErrorOutcomesNotStored, RpcReceiptToTxErrorWindowTooLarge, RpcReceiptToTxErrorMalformedHint, RpcReceiptToTxErrorBudgetExceeded]]):
    pass

