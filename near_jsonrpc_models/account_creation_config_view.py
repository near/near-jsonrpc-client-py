"""The structure describes configuration for creation of new accounts."""

from near_jsonrpc_models.account_id import AccountId
from pydantic import BaseModel
from pydantic import conint


class AccountCreationConfigView(BaseModel):
    # The minimum length of the top-level account ID that is allowed to be created by any account.
    min_allowed_top_level_account_length: conint(ge=0, le=255) = None
    # The account ID of the account registrar. This account ID allowed to create top-level
    # accounts of any valid length.
    registrar_account_id: AccountId = None
