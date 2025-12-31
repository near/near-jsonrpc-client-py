from models.public_key import PublicKey
from pydantic import BaseModel


class Tier1ProxyView(BaseModel):
    addr: str
    peer_id: PublicKey
