"""`ShardUId` is a unique representation for shards from different shard layouts.

Comparing to `ShardId`, which is just an ordinal number ranging from 0 to NUM_SHARDS-1,
`ShardUId` provides a way to unique identify shards when shard layouts may change across epochs.
This is important because we store states indexed by shards in our database, so we need a
way to unique identify shard even when shards change across epochs.
Another difference between `ShardUId` and `ShardId` is that `ShardUId` should only exist in
a node's internal state while `ShardId` can be exposed to outside APIs and used in protocol
level information (for example, `ShardChunkHeader` contains `ShardId` instead of `ShardUId`)"""

from pydantic import BaseModel
from pydantic import conint


class ShardUId(BaseModel):
    shard_id: conint(ge=0, le=4294967295)
    version: conint(ge=0, le=4294967295)
