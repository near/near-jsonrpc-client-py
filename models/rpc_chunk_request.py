from models.block_id import BlockId
from models.crypto_hash import CryptoHash
from models.shard_id import ShardId
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcChunkRequestBlockShardIdOption(BaseModel):
    block_id: BlockId
    shard_id: ShardId

class RpcChunkRequestChunkHashOption(BaseModel):
    chunk_id: CryptoHash

class RpcChunkRequest(RootModel[Union[RpcChunkRequestBlockShardIdOption, RpcChunkRequestChunkHashOption]]):
    pass

