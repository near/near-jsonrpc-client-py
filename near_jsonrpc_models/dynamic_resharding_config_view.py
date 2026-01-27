"""Configuration for dynamic resharding feature
See [`DynamicReshardingConfig`] for more details."""

from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel
from pydantic import conint
from typing import List


class DynamicReshardingConfigView(BaseModel):
    # Shards that should **not** be split even when they meet the regular split criteria.
    block_split_shards: List[ShardId]
    # Shards that should be split even when they don't meet the regular split criteria.
    force_split_shards: List[ShardId]
    # Maximum number of shards in the network.
    max_number_of_shards: conint(ge=0, le=18446744073709551615)
    # Memory threshold over which a shard is marked for a split.
    memory_usage_threshold: conint(ge=0, le=18446744073709551615)
    # Minimum memory usage of a child shard.
    min_child_memory_usage: conint(ge=0, le=18446744073709551615)
    # Minimum number of epochs until next resharding can be scheduled.
    min_epochs_between_resharding: conint(ge=0, le=18446744073709551615)
