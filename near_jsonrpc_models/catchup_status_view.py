"""Status of the [catchup](https://near.github.io/nearcore/architecture/how/sync.html#catchup) process"""

from near_jsonrpc_models.block_status_view import BlockStatusView
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import conint
from pydantic import field_validator
from typing import Dict
from typing import List


class CatchupStatusView(BaseModel):
    blocks_to_catchup: List[BlockStatusView]
    shard_sync_status: Dict[str, str]
    sync_block_hash: CryptoHash
    sync_block_height: conint(ge=0, le=18446744073709551615)

    @field_validator('shard_sync_status')
    def validate_shard_sync_status_keys(cls, v):
        import re
        pattern = re.compile(r"^\d+$")
        if not isinstance(v, dict):
            raise TypeError('shard_sync_status must be a dict')
        for key in v.keys():
            if not pattern.match(key):
                raise ValueError(f"Invalid key '{key}' in shard_sync_status. Must match '^\\d+$'")
        return v

