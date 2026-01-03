from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel
from pydantic import conint
from typing import List


class ShardLayoutV1(BaseModel):
    # The boundary accounts are the accounts on boundaries between shards.
    # Each shard contains a range of accounts from one boundary account to
    # another - or the smallest or largest account possible. The total
    # number of shards is equal to the number of boundary accounts plus 1.
    boundary_accounts: List[AccountId]
    # Maps shards from the last shard layout to shards that it splits to in this shard layout,
    # Useful for constructing states for the shards.
    # None for the genesis shard layout
    shards_split_map: List[List[ShardId]] | None = None
    # Maps shard in this shard layout to their parent shard
    # Since shard_ids always range from 0 to num_shards - 1, we use vec instead of a hashmap
    to_parent_shard_map: List[ShardId] | None = None
    # Version of the shard layout, this is useful for uniquely identify the shard layout
    version: conint(ge=0, le=4294967295)
