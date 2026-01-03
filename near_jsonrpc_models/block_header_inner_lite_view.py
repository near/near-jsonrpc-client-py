"""A part of a state for the current head of a light client. More info [here](https://nomicon.io/ChainSpec/LightClient)."""

from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import conint


class BlockHeaderInnerLiteView(BaseModel):
    # The merkle root of all the block hashes
    block_merkle_root: CryptoHash
    # The epoch to which the block that is the current known head belongs
    epoch_id: CryptoHash
    height: conint(ge=0, le=18446744073709551615)
    # The hash of the block producers set for the next epoch
    next_bp_hash: CryptoHash
    # The epoch that will follow the current epoch
    next_epoch_id: CryptoHash
    outcome_root: CryptoHash
    prev_state_root: CryptoHash
    # Legacy json number. Should not be used.
    timestamp: conint(ge=0, le=18446744073709551615)
    timestamp_nanosec: str
