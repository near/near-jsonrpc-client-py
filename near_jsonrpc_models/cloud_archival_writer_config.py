"""Configuration for a cloud-based archival writer. If this config is present, the writer is enabled and
writes chunk-related data based on the tracked shards. This config also controls additional archival
behavior such as block data and polling interval."""

from near_jsonrpc_models.duration_as_std_schema_provider import DurationAsStdSchemaProvider
from pydantic import BaseModel
from pydantic import Field


class CloudArchivalWriterConfig(BaseModel):
    # Determines whether block-related data should be written to cloud storage.
    archive_block_data: bool = False
    # Interval at which the system checks for new blocks or chunks to archive.
    polling_interval: DurationAsStdSchemaProvider = Field(default_factory=lambda: DurationAsStdSchemaProvider(**{'nanos': 0, 'secs': 1}))
