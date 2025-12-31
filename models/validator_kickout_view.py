from models.account_id import AccountId
from models.validator_kickout_reason import ValidatorKickoutReason
from pydantic import BaseModel


class ValidatorKickoutView(BaseModel):
    account_id: AccountId
    reason: ValidatorKickoutReason
