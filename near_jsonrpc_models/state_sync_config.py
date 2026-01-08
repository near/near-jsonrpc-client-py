from near_jsonrpc_models.dump_config import DumpConfig
from near_jsonrpc_models.sync_concurrency import SyncConcurrency
from near_jsonrpc_models.sync_config import SyncConfig
from pydantic import BaseModel
from pydantic import conint


class StateSyncConfig(BaseModel):
    concurrency: SyncConcurrency = None
    # `none` value disables state dump to external storage.
    dump: DumpConfig | None = None
    # Zstd compression level for state parts.
    parts_compression_lvl: conint(ge=-2147483648, le=2147483647) = 1
    sync: SyncConfig = None
