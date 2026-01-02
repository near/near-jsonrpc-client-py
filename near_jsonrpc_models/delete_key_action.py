from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel


class DeleteKeyAction(BaseModel):
    # A public key associated with the access_key to be deleted.
    public_key: PublicKey
