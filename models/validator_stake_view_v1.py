from models.account_id import AccountId
from models.near_token import NearToken
from models.public_key import PublicKey
from pydantic import BaseModel


class ValidatorStakeViewV1(BaseModel):
    account_id: AccountId
    public_key: PublicKey
    stake: NearToken
