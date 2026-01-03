"""A view of the account"""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.near_token import NearToken
from pydantic import BaseModel
from pydantic import conint


class RpcViewAccountResponse(BaseModel):
    amount: NearToken
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    code_hash: CryptoHash
    global_contract_account_id: AccountId | None = None
    global_contract_hash: CryptoHash | None = None
    locked: NearToken
    # TODO(2271): deprecated.
    storage_paid_at: conint(ge=0, le=18446744073709551615) = 0
    storage_usage: conint(ge=0, le=18446744073709551615)
