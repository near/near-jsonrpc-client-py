"""Account ID with its public key."""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel


class AccountWithPublicKey(BaseModel):
    account_id: AccountId
    public_key: PublicKey
