"""Resulting struct represents block with chunks"""

from near_jsonrpc_models.block_view import BlockView
from near_jsonrpc_models.indexer_shard import IndexerShard
from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel
from typing import List


class RpcIndexerBlockResponse(BaseModel):
    block: BlockView
    shards: List[IndexerShard]
    # The node's configured chunk and execution coverage, in block layout order.
    # Carried chunks remain None inside the message even for tracked shards.
    tracked_shards: List[ShardId]
