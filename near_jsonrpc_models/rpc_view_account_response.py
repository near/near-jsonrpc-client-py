"""A view of the account"""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.near_token import NearToken
from pydantic import BaseModel
from pydantic import conint


class RpcViewAccountResponse(BaseModel):
    # Liquid (non-staked) account balance, in yoctoNEAR.
    amount: NearToken
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    # Hash of the deployed contract code; the all-`1`s hash when no contract is deployed.
    code_hash: CryptoHash
    # Set when the account uses a global contract referenced by the deploying account id.
    global_contract_account_id: AccountId | None = None
    # Set when the account uses a global contract referenced by code hash.
    global_contract_hash: CryptoHash | None = None
    # Staked balance locked for validation, in yoctoNEAR.
    locked: NearToken
    # Deprecated and unused. TODO(2271): remove.
    storage_paid_at: conint(ge=0, le=18446744073709551615) = 0
    # Total storage used by the account, in bytes.
    storage_usage: conint(ge=0, le=18446744073709551615)
