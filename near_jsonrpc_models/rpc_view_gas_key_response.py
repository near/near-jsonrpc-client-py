from near_jsonrpc_models.access_key_permission_view import AccessKeyPermissionView
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.near_token import NearToken
from pydantic import BaseModel
from pydantic import conint
from typing import List


class RpcViewGasKeyResponse(BaseModel):
    balance: NearToken
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    nonces: List[conint(ge=0, le=18446744073709551615)]
    num_nonces: conint(ge=0, le=4294967295)
    permission: AccessKeyPermissionView
