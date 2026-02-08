"""Contains main info about the block."""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.shard_id import ShardId
from near_jsonrpc_models.signature import Signature
from near_jsonrpc_models.slashed_validator import SlashedValidator
from near_jsonrpc_models.validator_stake_view import ValidatorStakeView
from pydantic import BaseModel
from pydantic import Field
from pydantic import conint
from typing import List
from typing import Tuple


class BlockHeaderView(BaseModel):
    approvals: List[Signature | None]
    block_body_hash: CryptoHash | None = None
    block_merkle_root: CryptoHash
    block_ordinal: conint(ge=0, le=18446744073709551615) | None = None
    challenges_result: List[SlashedValidator]
    challenges_root: CryptoHash
    chunk_endorsements: List[List[conint(ge=0, le=255)]] | None = None
    chunk_headers_root: CryptoHash
    chunk_mask: List[bool]
    chunk_receipts_root: CryptoHash
    chunk_tx_root: CryptoHash
    chunks_included: conint(ge=0, le=18446744073709551615)
    epoch_id: CryptoHash
    epoch_sync_data_hash: CryptoHash | None = None
    gas_price: NearToken
    hash: CryptoHash
    height: conint(ge=0, le=18446744073709551615)
    last_ds_final_block: CryptoHash
    last_final_block: CryptoHash
    latest_protocol_version: conint(ge=0, le=4294967295)
    next_bp_hash: CryptoHash
    next_epoch_id: CryptoHash
    outcome_root: CryptoHash
    # The hash of the previous Block
    prev_hash: CryptoHash
    prev_height: conint(ge=0, le=18446744073709551615) | None = None
    prev_state_root: CryptoHash
    random_value: CryptoHash
    # TODO(2271): deprecated.
    rent_paid: NearToken = Field(default_factory=lambda: NearToken('0'))
    shard_split: Tuple[ShardId, AccountId] | None = None
    # Signature of the block producer.
    signature: Signature
    # Legacy json number. Should not be used.
    timestamp: conint(ge=0, le=18446744073709551615)
    timestamp_nanosec: str
    total_supply: NearToken
    validator_proposals: List[ValidatorStakeView]
    # TODO(2271): deprecated.
    validator_reward: NearToken = Field(default_factory=lambda: NearToken('0'))
