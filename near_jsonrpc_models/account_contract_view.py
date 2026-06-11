"""RPC view of a non-empty [`AccountContract`]. The `AccountContract::None`
variant is represented externally as a JSON `null` via `Option`, so this
enum only carries the three "contract is present" cases. Serializes as
an externally-tagged object:

- `Local(hash)` → `{"local": "<CryptoHash>"}`
- `GlobalHash(hash)` → `{"global_hash": "<CryptoHash>"}`
- `GlobalAccountId(id)` → `{"global_account_id": "<AccountId>"}`

Mirrors [`AccountContract`] 1:1 (minus `None`) so consumers can preserve
the distinction between a global-by-hash and global-by-account contract
without descending into a nested identifier."""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class AccountContractViewLocal(StrictBaseModel):
    local: CryptoHash

class AccountContractViewGlobalHash(StrictBaseModel):
    global_hash: CryptoHash

class AccountContractViewGlobalAccountId(StrictBaseModel):
    global_account_id: AccountId

class AccountContractView(RootModel[Union[AccountContractViewLocal, AccountContractViewGlobalHash, AccountContractViewGlobalAccountId]]):
    pass

