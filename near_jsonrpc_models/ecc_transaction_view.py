from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel


class EccTransactionView(BaseModel):
    contract_id: AccountId
    payload: str
    # The universal account state init this copy of the call carries.
    state_init: str | None = None
    # The unique id of this copy of the call, its outcome id.
    transaction_id: CryptoHash
