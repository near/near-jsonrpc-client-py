"""Gas key is like an access key, except it stores a balance separately, and transactions signed
with it deduct their cost from the gas key balance instead of the account balance."""

from near_jsonrpc_models.access_key_permission import AccessKeyPermission
from near_jsonrpc_models.near_token import NearToken
from pydantic import BaseModel
from pydantic import conint


class GasKey(BaseModel):
    # The balance of the gas key.
    balance: NearToken
    # The number of nonces this gas key has.
    num_nonces: conint(ge=0, le=4294967295)
    # Defines the permissions for this gas key.
    # If this is a `FunctionCallPermission`, the allowance must be None (unlimited).
    permission: AccessKeyPermission
