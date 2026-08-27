"""Create a `0u` universal account from its state init. The receiver id must
equal `derive_universal_account_id(state_init)`; the attached `deposit`
covers the new account's storage staking.

The state init travels as the bytes the producer serialized, because the
receiver id commits to exactly those bytes. The typed [`UniversalStateInit`]
is a decoded view of them, used where the state has to be installed or priced."""

from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.raw_state_init import RawStateInit
from pydantic import BaseModel


class UniversalStateInitAction(BaseModel):
    deposit: NearToken
    state_init: RawStateInit
