from near_jsonrpc_models.duration_as_std_schema_provider import DurationAsStdSchemaProvider
from pydantic import BaseModel
from pydantic import conint


class EpochSyncConfig(BaseModel):
    # Number of epochs behind the network head beyond which the node will use
    # epoch sync instead of header sync. Also the maximum age (in epochs) of
    # accepted epoch sync proofs. At the consumption site, this is multiplied
    # by epoch_length to get the horizon in blocks.
    epoch_sync_horizon_num_epochs: conint(ge=0, le=18446744073709551615) = 4
    # Timeout for epoch sync requests. The node will continue retrying indefinitely even
    # if this timeout is exceeded.
    timeout_for_epoch_sync: DurationAsStdSchemaProvider = None
