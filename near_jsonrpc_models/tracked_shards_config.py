"""Describes the expected behavior of the node regarding shard tracking.
If the node is an active validator, it will also track the shards it is responsible for as a validator."""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.shard_id import ShardId
from near_jsonrpc_models.shard_uid import ShardUId
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import List
from typing import Literal
from typing import Union


"""Tracks no shards (light client)."""
class TrackedShardsConfigNoShards(RootModel[Literal['NoShards']]):
    pass

class TrackedShardsConfigShards(StrictBaseModel):
    """Tracks arbitrary shards."""
    Shards: List[ShardUId]

"""Tracks all shards."""
class TrackedShardsConfigAllShards(RootModel[Literal['AllShards']]):
    pass

class TrackedShardsConfigShadowValidator(StrictBaseModel):
    """Tracks shards that are assigned to given validator account."""
    ShadowValidator: AccountId

class TrackedShardsConfigSchedule(StrictBaseModel):
    """Rotate between these sets of tracked shards.
Used to simulate the behavior of chunk only producers without staking tokens."""
    Schedule: List[List[ShardId]]

class TrackedShardsConfigAccounts(StrictBaseModel):
    """Tracks shards that contain one of the given account."""
    Accounts: List[AccountId]

class TrackedShardsConfig(RootModel[Union[TrackedShardsConfigNoShards, TrackedShardsConfigShards, TrackedShardsConfigAllShards, TrackedShardsConfigShadowValidator, TrackedShardsConfigSchedule, TrackedShardsConfigAccounts]]):
    pass

