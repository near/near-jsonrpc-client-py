"""The shard identifier. It may be an arbitrary number - it does not need to be
a number in the range 0..NUM_SHARDS. The shard ids do not need to be
sequential or contiguous.

The shard id is wrapped in a new type to prevent the old pattern of using
indices in range 0..NUM_SHARDS and casting to ShardId. Once the transition
if fully complete it potentially may be simplified to a regular type alias."""

from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint


class ShardId(RootModel[conint(ge=0, le=18446744073709551615)]):
    pass

