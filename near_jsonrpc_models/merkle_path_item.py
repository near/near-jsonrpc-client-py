from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.direction import Direction
from pydantic import BaseModel


class MerklePathItem(BaseModel):
    direction: Direction
    hash: CryptoHash
