"""A view of the account"""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.account_state import AccountState
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.near_token import NearToken
from pydantic import BaseModel
from pydantic import conint


class RpcViewAccountResponse(BaseModel):
    # Liquid (non-staked) account balance, in yoctoNEAR.
    amount: NearToken
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    # The nonce an uninitialized account's own transactions must use, present
    # only while it is uninitialized. A self-signed state init is the one
    # transaction such an account can send, and this is the only way for a
    # client to learn the nonce it must carry: there is no access key to query.
    bootstrap_nonce: conint(ge=0, le=18446744073709551615) | None = None
    # Hash of the deployed contract code; the all-`1`s hash when no contract is deployed.
    code_hash: CryptoHash
    # Set when the account uses a global contract referenced by the deploying account id.
    global_contract_account_id: AccountId | None = None
    # Set when the account uses a global contract referenced by code hash.
    global_contract_hash: CryptoHash | None = None
    # Staked balance locked for validation, in yoctoNEAR.
    locked: NearToken
    # Whether the account is initialized. Only a universal account can be
    # uninitialized: it has no access keys, code or data until a
    # `UniversalStateInit` arrives. Omitted for initialized accounts.
    state: AccountState = None
    # Deprecated and unused. TODO(2271): remove.
    storage_paid_at: conint(ge=0, le=18446744073709551615) = 0
    # Total storage used by the account, in bytes.
    storage_usage: conint(ge=0, le=18446744073709551615)
