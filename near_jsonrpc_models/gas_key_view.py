from near_jsonrpc_models.access_key_permission_view import AccessKeyPermissionView
from near_jsonrpc_models.near_token import NearToken
from pydantic import BaseModel
from pydantic import conint
from typing import List


class GasKeyView(BaseModel):
    balance: NearToken
    nonces: List[conint(ge=0, le=18446744073709551615)]
    num_nonces: conint(ge=0, le=4294967295)
    permission: AccessKeyPermissionView
