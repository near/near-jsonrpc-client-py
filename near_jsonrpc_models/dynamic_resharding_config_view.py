"""Configuration for dynamic resharding feature"""

from pydantic import BaseModel
from pydantic import conint


class DynamicReshardingConfigView(BaseModel):
    # Maximum number of shards in the network.
    # 
    # See [`CongestionControlConfig`] for more details.
    max_number_of_shards: conint(ge=0, le=18446744073709551615)
    # Memory threshold over which a shard is marked for a split.
    # 
    # See [`CongestionControlConfig`] for more details.
    memory_usage_threshold: conint(ge=0, le=18446744073709551615)
    # Minimum memory usage of a child shard.
    # 
    # See [`CongestionControlConfig`] for more details.
    min_child_memory_usage: conint(ge=0, le=18446744073709551615)
    # Minimum number of epochs until next resharding can be scheduled.
    # 
    # See [`CongestionControlConfig`] for more details.
    min_epochs_between_resharding: conint(ge=0, le=18446744073709551615)
