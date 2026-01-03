from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcCongestionLevelRequestBlockShardId(BaseModel):
    block_id: BlockId
    shard_id: ShardId

class RpcCongestionLevelRequestChunkHash(BaseModel):
    chunk_id: CryptoHash

class RpcCongestionLevelRequest(RootModel[Union[RpcCongestionLevelRequestBlockShardId, RpcCongestionLevelRequestChunkHash]]):
    pass

