"""Configures how to dump state to external storage."""

from near_jsonrpc_models.duration_as_std_schema_provider import DurationAsStdSchemaProvider
from near_jsonrpc_models.external_storage_location import ExternalStorageLocation
from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel
from typing import List


class DumpConfig(BaseModel):
    # Location of a json file with credentials allowing access to the bucket.
    credentials_file: str | None = None
    # How often to check if a new epoch has started.
    # Feel free to set to `None`, defaults are sensible.
    iteration_delay: DurationAsStdSchemaProvider | None = None
    # Specifies where to write the obtained state parts.
    location: ExternalStorageLocation = None
    # Use in case a node that dumps state to the external storage
    # gets in trouble.
    restart_dump_for_shards: List[ShardId] | None = None
