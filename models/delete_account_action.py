from models.account_id import AccountId
from pydantic import BaseModel


class DeleteAccountAction(BaseModel):
    beneficiary_id: AccountId
