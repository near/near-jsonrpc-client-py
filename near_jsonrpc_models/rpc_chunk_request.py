from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcChunkRequestBlockShardId(BaseModel):
    block_id: BlockId
    shard_id: ShardId

class RpcChunkRequestChunkHash(BaseModel):
    chunk_id: CryptoHash

class RpcChunkRequest(RootModel[Union[RpcChunkRequestBlockShardId, RpcChunkRequestChunkHash]]):
    pass

