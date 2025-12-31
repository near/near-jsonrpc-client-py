"""Account info for validators"""

from models.account_id import AccountId
from models.near_token import NearToken
from models.public_key import PublicKey
from pydantic import BaseModel


class AccountInfo(BaseModel):
    account_id: AccountId
    amount: NearToken
    public_key: PublicKey
