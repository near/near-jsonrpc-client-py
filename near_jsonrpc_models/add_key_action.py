"""An action that adds key with public key associated"""

from near_jsonrpc_models.access_key import AccessKey
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel


class AddKeyAction(BaseModel):
    # An access key with the permission
    access_key: AccessKey
    # A public key which will be associated with an access_key
    public_key: PublicKey
