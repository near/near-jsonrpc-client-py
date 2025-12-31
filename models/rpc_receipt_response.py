from models.account_id import AccountId
from models.crypto_hash import CryptoHash
from models.receipt_enum_view import ReceiptEnumView
from pydantic import BaseModel
from pydantic import conint


class RpcReceiptResponse(BaseModel):
    predecessor_id: AccountId
    priority: conint(ge=0, le=18446744073709551615) = 0
    receipt: ReceiptEnumView
    receipt_id: CryptoHash
    receiver_id: AccountId
