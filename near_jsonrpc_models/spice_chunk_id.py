"""In spice missing chunks and equivalent to empty chunks so block hash and shard id always
uniquely identifies chunks."""

from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel


class SpiceChunkId(BaseModel):
    block_hash: CryptoHash
    shard_id: ShardId
