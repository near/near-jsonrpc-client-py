from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.epoch_id import EpochId
from pydantic import BaseModel
from pydantic import conint


class StatusSyncInfo(BaseModel):
    earliest_block_hash: CryptoHash | None = None
    earliest_block_height: conint(ge=0, le=18446744073709551615) | None = None
    earliest_block_time: str | None = None
    epoch_id: EpochId | None = None
    epoch_start_height: conint(ge=0, le=18446744073709551615) | None = None
    latest_block_hash: CryptoHash
    latest_block_height: conint(ge=0, le=18446744073709551615)
    latest_block_time: str
    latest_state_root: CryptoHash
    syncing: bool
