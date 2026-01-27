from near_jsonrpc_models.account_id import AccountId
from pydantic import BaseModel


class SlashedValidator(BaseModel):
    account_id: AccountId
    is_double_sign: bool
