"""Account ID with its public key."""

from models.account_id import AccountId
from models.public_key import PublicKey
from pydantic import BaseModel


class AccountWithPublicKey(BaseModel):
    account_id: AccountId
    public_key: PublicKey
