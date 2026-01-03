"""Describes information about the current epoch validator"""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel
from pydantic import Field
from pydantic import conint
from typing import List


class CurrentEpochValidatorInfo(BaseModel):
    account_id: AccountId
    is_slashed: bool
    num_expected_blocks: conint(ge=0, le=18446744073709551615)
    num_expected_chunks: conint(ge=0, le=18446744073709551615) = 0
    # Number of chunks this validator was expected to produce in each shard.
    # Each entry in the array corresponds to the shard in the `shards_produced` array.
    num_expected_chunks_per_shard: List[conint(ge=0, le=18446744073709551615)] = Field(default_factory=lambda: [])
    num_expected_endorsements: conint(ge=0, le=18446744073709551615) = 0
    # Number of chunks this validator was expected to validate and endorse in each shard.
    # Each entry in the array corresponds to the shard in the `shards_endorsed` array.
    num_expected_endorsements_per_shard: List[conint(ge=0, le=18446744073709551615)] = Field(default_factory=lambda: [])
    num_produced_blocks: conint(ge=0, le=18446744073709551615)
    num_produced_chunks: conint(ge=0, le=18446744073709551615) = 0
    num_produced_chunks_per_shard: List[conint(ge=0, le=18446744073709551615)] = Field(default_factory=lambda: [])
    num_produced_endorsements: conint(ge=0, le=18446744073709551615) = 0
    num_produced_endorsements_per_shard: List[conint(ge=0, le=18446744073709551615)] = Field(default_factory=lambda: [])
    public_key: PublicKey
    # Shards this validator is assigned to as chunk producer in the current epoch.
    shards: List[ShardId]
    # Shards this validator is assigned to as chunk validator in the current epoch.
    shards_endorsed: List[ShardId] = Field(default_factory=lambda: [ShardId(**item) for item in []])
    stake: NearToken
