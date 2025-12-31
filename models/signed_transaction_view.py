from models.account_id import AccountId
from models.action_view import ActionView
from models.crypto_hash import CryptoHash
from models.public_key import PublicKey
from models.signature import Signature
from pydantic import BaseModel
from pydantic import conint
from typing import List


class SignedTransactionView(BaseModel):
    actions: List[ActionView]
    hash: CryptoHash
    nonce: conint(ge=0, le=18446744073709551615)
    priority_fee: conint(ge=0, le=18446744073709551615) = 0
    public_key: PublicKey
    receiver_id: AccountId
    signature: Signature
    signer_id: AccountId
