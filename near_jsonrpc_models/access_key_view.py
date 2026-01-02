"""Describes access key permission scope and nonce."""

from near_jsonrpc_models.access_key_permission_view import AccessKeyPermissionView
from pydantic import BaseModel
from pydantic import conint


class AccessKeyView(BaseModel):
    nonce: conint(ge=0, le=18446744073709551615)
    permission: AccessKeyPermissionView
