"""Configuration for a cloud-based archival writer. If this config is present, the writer is enabled and
writes chunk-related data based on the tracked shards. This config also controls additional archival
behavior such as block data and polling interval."""

from near_jsonrpc_models.duration_as_std_schema_provider import DurationAsStdSchemaProvider
from pydantic import BaseModel
from pydantic import Field
from pydantic import conint


class CloudArchivalWriterConfig(BaseModel):
    # Determines whether block-related data should be written to cloud storage.
    archive_block_data: bool = False
    # Interval at which the system checks for new blocks or chunks to archive.
    polling_interval: DurationAsStdSchemaProvider = Field(default_factory=lambda: DurationAsStdSchemaProvider(**{'nanos': 0, 'secs': 1}))
    # Cadence of state snapshots, in epochs. Higher values reduce bucket cost at
    # the expense of potentially longer delta replay during reader bootstrap.
    snapshot_every_n_epochs: conint(ge=0, le=18446744073709551615) = 10
