from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel


class EccTransactionView(BaseModel):
    contract_id: AccountId
    hash: CryptoHash
    payload: str
