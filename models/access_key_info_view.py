"""Describes information about an access key including the public key."""

from models.access_key_view import AccessKeyView
from models.public_key import PublicKey
from pydantic import BaseModel


class AccessKeyInfoView(BaseModel):
    access_key: AccessKeyView
    public_key: PublicKey
