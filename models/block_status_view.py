"""Height and hash of a block"""

from models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import conint


class BlockStatusView(BaseModel):
    hash: CryptoHash
    height: conint(ge=0, le=18446744073709551615)
