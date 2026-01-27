from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.receipt_enum_view import ReceiptEnumView
from pydantic import BaseModel
from pydantic import conint


class RpcReceiptResponse(BaseModel):
    predecessor_id: AccountId
    # Deprecated, retained for backward compatibility.
    priority: conint(ge=0, le=18446744073709551615) = 0
    receipt: ReceiptEnumView
    receipt_id: CryptoHash
    receiver_id: AccountId
