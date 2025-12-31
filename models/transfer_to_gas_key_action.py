from models.near_token import NearToken
from models.public_key import PublicKey
from pydantic import BaseModel


class TransferToGasKeyAction(BaseModel):
    deposit: NearToken
    public_key: PublicKey
