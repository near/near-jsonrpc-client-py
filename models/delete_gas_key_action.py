from models.public_key import PublicKey
from pydantic import BaseModel


class DeleteGasKeyAction(BaseModel):
    public_key: PublicKey
