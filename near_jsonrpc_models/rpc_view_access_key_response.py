"""Describes access key permission scope and nonce."""

from near_jsonrpc_models.access_key_permission_view import AccessKeyPermissionView
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import conint


class RpcViewAccessKeyResponse(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    # Current nonce; each transaction signed with this key must use a strictly greater value.
    nonce: conint(ge=0, le=18446744073709551615)
    # Access scope: full access, or a function-call permission with an optional allowance and method/receiver limits.
    permission: AccessKeyPermissionView
