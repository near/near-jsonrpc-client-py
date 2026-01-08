"""Describes access key permission scope and nonce."""

from near_jsonrpc_models.access_key_permission_view import AccessKeyPermissionView
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import conint


class RpcViewAccessKeyResponse(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    nonce: conint(ge=0, le=18446744073709551615)
    permission: AccessKeyPermissionView
