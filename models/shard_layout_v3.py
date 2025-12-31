"""Counterpart to `ShardLayoutV3` composed of maps with string keys to aid
serde serialization."""

from models.account_id import AccountId
from models.shard_id import ShardId
from pydantic import BaseModel
from pydantic import conint
from typing import Dict
from typing import List


class ShardLayoutV3(BaseModel):
    boundary_accounts: List[AccountId]
    id_to_index_map: Dict[str, conint(ge=0, le=4294967295)]
    last_split: ShardId
    shard_ids: List[ShardId]
    shards_split_map: Dict[str, List[ShardId]]
