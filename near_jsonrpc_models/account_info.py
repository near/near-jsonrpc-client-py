"""Account info for validators"""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel


class AccountInfo(BaseModel):
    account_id: AccountId
    amount: NearToken
    public_key: PublicKey
