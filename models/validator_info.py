from models.account_id import AccountId
from pydantic import BaseModel


class ValidatorInfo(BaseModel):
    account_id: AccountId
