from near_jsonrpc_models.duration_as_std_schema_provider import DurationAsStdSchemaProvider
from pydantic import BaseModel
from pydantic import conint


class EpochSyncConfig(BaseModel):
    # This serves as two purposes: (1) the node will not epoch sync and instead resort to
    # header sync, if the genesis block is within this many blocks from the current block;
    # (2) the node will reject an epoch sync proof if the provided proof is for an epoch
    # that is more than this many blocks behind the current block.
    epoch_sync_horizon: conint(ge=0, le=18446744073709551615) = None
    # Timeout for epoch sync requests. The node will continue retrying indefinitely even
    # if this timeout is exceeded.
    timeout_for_epoch_sync: DurationAsStdSchemaProvider = None
