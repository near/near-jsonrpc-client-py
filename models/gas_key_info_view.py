from models.gas_key_view import GasKeyView
from models.public_key import PublicKey
from pydantic import BaseModel


class GasKeyInfoView(BaseModel):
    gas_key: GasKeyView
    public_key: PublicKey
