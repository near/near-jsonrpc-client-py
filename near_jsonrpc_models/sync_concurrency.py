from pydantic import BaseModel
from pydantic import conint


class SyncConcurrency(BaseModel):
    # Maximum number of "apply parts" tasks that can be performed in parallel.
    # This is a very disk-heavy task and therefore we set this to a low limit,
    # or else the rocksdb contention makes the whole server freeze up.
    apply: conint(ge=0, le=255) = None
    # Maximum number of "apply parts" tasks that can be performed in parallel
    # during catchup. We set this to a very low value to avoid overloading the
    # node while it is still performing normal tasks.
    apply_during_catchup: conint(ge=0, le=255) = None
    # Maximum number of outstanding requests for decentralized state sync.
    peer_downloads: conint(ge=0, le=255) = None
    # The maximum parallelism to use per shard. This is mostly for fairness, because
    # the actual rate limiting is done by the TaskTrackers, but this is useful for
    # balancing the shards a little.
    per_shard: conint(ge=0, le=255) = None
