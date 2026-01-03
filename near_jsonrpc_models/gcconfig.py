"""Configuration for garbage collection."""

from near_jsonrpc_models.duration_as_std_schema_provider import DurationAsStdSchemaProvider
from pydantic import BaseModel
from pydantic import Field
from pydantic import conint


class GCConfig(BaseModel):
    # Maximum number of blocks to garbage collect at every garbage collection
    # call.
    gc_blocks_limit: conint(ge=0, le=18446744073709551615) = 2
    # Maximum number of height to go through at each garbage collection step
    # when cleaning forks during garbage collection.
    gc_fork_clean_step: conint(ge=0, le=18446744073709551615) = 100
    # Number of epochs for which we keep store data.
    gc_num_epochs_to_keep: conint(ge=0, le=18446744073709551615) = 5
    # How often gc should be run
    gc_step_period: DurationAsStdSchemaProvider = Field(default_factory=lambda: DurationAsStdSchemaProvider(**{'nanos': 500000000, 'secs': 0}))
