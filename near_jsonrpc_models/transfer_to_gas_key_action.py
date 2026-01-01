from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel


class TransferToGasKeyAction(BaseModel):
    deposit: NearToken
    public_key: PublicKey
