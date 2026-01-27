from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.action_view import ActionView
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.signature import Signature
from pydantic import BaseModel
from pydantic import conint
from typing import List


class SignedTransactionView(BaseModel):
    actions: List[ActionView]
    hash: CryptoHash
    nonce: conint(ge=0, le=18446744073709551615)
    nonce_index: conint(ge=0, le=4294967295) | None = None
    # Deprecated, retained for backward compatibility.
    priority_fee: conint(ge=0, le=18446744073709551615) = 0
    public_key: PublicKey
    receiver_id: AccountId
    signature: Signature
    signer_id: AccountId
