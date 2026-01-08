"""Describes information about an access key including the public key."""

from near_jsonrpc_models.access_key_view import AccessKeyView
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel


class AccessKeyInfoView(BaseModel):
    access_key: AccessKeyView
    public_key: PublicKey
