"""Contains main info about the chunk."""

from near_jsonrpc_models.bandwidth_requests import BandwidthRequests
from near_jsonrpc_models.congestion_info_view import CongestionInfoView
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.near_gas import NearGas
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.shard_id import ShardId
from near_jsonrpc_models.signature import Signature
from near_jsonrpc_models.validator_stake_view import ValidatorStakeView
from pydantic import BaseModel
from pydantic import Field
from pydantic import conint
from typing import List


class ChunkHeaderView(BaseModel):
    balance_burnt: NearToken
    bandwidth_requests: BandwidthRequests | None = None
    chunk_hash: CryptoHash
    congestion_info: CongestionInfoView | None = None
    encoded_length: conint(ge=0, le=18446744073709551615)
    encoded_merkle_root: CryptoHash
    gas_limit: NearGas
    gas_used: NearGas
    height_created: conint(ge=0, le=18446744073709551615)
    height_included: conint(ge=0, le=18446744073709551615)
    outcome_root: CryptoHash
    outgoing_receipts_root: CryptoHash
    prev_block_hash: CryptoHash
    prev_state_root: CryptoHash
    # TODO(2271): deprecated.
    rent_paid: NearToken = Field(default_factory=lambda: NearToken('0'))
    shard_id: ShardId
    signature: Signature
    tx_root: CryptoHash
    validator_proposals: List[ValidatorStakeView]
    # TODO(2271): deprecated.
    validator_reward: NearToken = Field(default_factory=lambda: NearToken('0'))
