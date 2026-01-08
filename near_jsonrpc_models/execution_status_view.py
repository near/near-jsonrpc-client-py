from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.strict_model import StrictBaseModel
from near_jsonrpc_models.tx_execution_error import TxExecutionError
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


"""The execution is pending or unknown."""
class ExecutionStatusViewUnknown(RootModel[Literal['Unknown']]):
    pass

class ExecutionStatusViewFailure(StrictBaseModel):
    """The execution has failed."""
    Failure: TxExecutionError

class ExecutionStatusViewSuccessValue(StrictBaseModel):
    """The final action succeeded and returned some value or an empty vec encoded in base64."""
    SuccessValue: str

class ExecutionStatusViewSuccessReceiptId(StrictBaseModel):
    """The final action of the receipt returned a promise or the signed transaction was converted
to a receipt. Contains the receipt_id of the generated receipt."""
    SuccessReceiptId: CryptoHash

class ExecutionStatusView(RootModel[Union[ExecutionStatusViewUnknown, ExecutionStatusViewFailure, ExecutionStatusViewSuccessValue, ExecutionStatusViewSuccessReceiptId]]):
    pass

