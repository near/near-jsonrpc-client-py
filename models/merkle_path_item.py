from models.crypto_hash import CryptoHash
from models.direction import Direction
from pydantic import BaseModel


class MerklePathItem(BaseModel):
    direction: Direction
    hash: CryptoHash
