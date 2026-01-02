from datetime import datetime
from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.account_info import AccountInfo
from near_jsonrpc_models.near_gas import NearGas
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.shard_layout import ShardLayout
from pydantic import BaseModel
from pydantic import Field
from pydantic import conint
from pydantic import conlist
from pydantic import field_validator
from typing import List


class GenesisConfig(BaseModel):
    # Expected number of hidden validators per shard.
    avg_hidden_validator_seats_per_shard: List[conint(ge=0, le=18446744073709551615)]
    # Threshold for kicking out block producers, between 0 and 100.
    block_producer_kickout_threshold: conint(ge=0, le=255)
    # ID of the blockchain. This must be unique for every blockchain.
    # If your testnet blockchains do not have unique chain IDs, you will have a bad time.
    chain_id: str
    # Limits the number of shard changes in chunk producer assignments,
    # if algorithm is able to choose assignment with better balance of
    # number of chunk producers for shards.
    chunk_producer_assignment_changes_limit: conint(ge=0, le=18446744073709551615) = 5
    # Threshold for kicking out chunk producers, between 0 and 100.
    chunk_producer_kickout_threshold: conint(ge=0, le=255)
    # Threshold for kicking out nodes which are only chunk validators, between 0 and 100.
    chunk_validator_only_kickout_threshold: conint(ge=0, le=255) = 80
    # Enable dynamic re-sharding.
    dynamic_resharding: bool
    # Epoch length counted in block heights.
    epoch_length: conint(ge=0, le=18446744073709551615)
    # Fishermen stake threshold.
    fishermen_threshold: NearToken
    # Initial gas limit.
    gas_limit: NearGas
    # Gas price adjustment rate
    gas_price_adjustment_rate: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2)
    # Height of genesis block.
    genesis_height: conint(ge=0, le=18446744073709551615)
    # Official time of blockchain start.
    genesis_time: datetime
    max_gas_price: NearToken
    # Maximum inflation on the total supply every epoch.
    max_inflation_rate: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2)
    # Max stake percentage of the validators we will kick out.
    max_kickout_stake_perc: conint(ge=0, le=255) = 100
    # Minimum gas price. It is also the initial gas price.
    min_gas_price: NearToken
    # The minimum stake required for staking is last seat price divided by this number.
    minimum_stake_divisor: conint(ge=0, le=18446744073709551615) = 10
    # The lowest ratio s/s_total any block producer can have.
    # See <https://github.com/near/NEPs/pull/167> for details
    minimum_stake_ratio: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = Field(default_factory=lambda: [1, 6250])
    # The minimum number of validators each shard must have
    minimum_validators_per_shard: conint(ge=0, le=18446744073709551615) = 1
    # Number of block producer seats at genesis.
    num_block_producer_seats: conint(ge=0, le=18446744073709551615)
    # Defines number of shards and number of block producer seats per each shard at genesis.
    # Note: not used with protocol_feature_chunk_only_producers -- replaced by minimum_validators_per_shard
    # Note: not used before as all block producers produce chunks for all shards
    num_block_producer_seats_per_shard: List[conint(ge=0, le=18446744073709551615)]
    # Expected number of blocks per year
    num_blocks_per_year: conint(ge=0, le=18446744073709551615)
    # Deprecated.
    num_chunk_only_producer_seats: conint(ge=0, le=18446744073709551615) = 300
    # Number of chunk producers.
    # Don't mess it up with chunk-only producers feature which is deprecated.
    num_chunk_producer_seats: conint(ge=0, le=18446744073709551615) = 100
    num_chunk_validator_seats: conint(ge=0, le=18446744073709551615) = 300
    # Online maximum threshold above which validator gets full reward.
    online_max_threshold: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = Field(default_factory=lambda: [99, 100])
    # Online minimum threshold below which validator doesn't receive reward.
    online_min_threshold: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = Field(default_factory=lambda: [9, 10])
    # Protocol treasury rate
    protocol_reward_rate: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2)
    # Protocol treasury account
    protocol_treasury_account: AccountId
    # Threshold of stake that needs to indicate that they ready for upgrade.
    protocol_upgrade_stake_threshold: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = Field(default_factory=lambda: [4, 5])
    # Protocol version that this genesis works with.
    protocol_version: conint(ge=0, le=4294967295)
    # Layout information regarding how to split accounts to shards
    shard_layout: ShardLayout = Field(default_factory=lambda: ShardLayout(**{'V2': {'boundary_accounts': [], 'id_to_index_map': {'0': 0}, 'index_to_id_map': {'0': 0}, 'shard_ids': [0], 'version': 0}}))
    # If true, shuffle the chunk producers across shards. In other words, if
    # the shard assignments were `[S_0, S_1, S_2, S_3]` where `S_i` represents
    # the set of chunk producers for shard `i`, if this flag were true, the
    # shard assignments might become, for example, `[S_2, S_0, S_3, S_1]`.
    shuffle_shard_assignment_for_chunk_producers: bool = False
    # Number of target chunk validator mandates for each shard.
    target_validator_mandates_per_shard: conint(ge=0, le=18446744073709551615) = 68
    # Total supply of tokens at genesis.
    total_supply: NearToken
    # Number of blocks for which a given transaction is valid
    transaction_validity_period: conint(ge=0, le=18446744073709551615)
    # This is only for test purposes. We hard code some configs for mainnet and testnet
    # in AllEpochConfig, and we want to have a way to test that code path. This flag is for that.
    # If set to true, the node will use the same config override path as mainnet and testnet.
    use_production_config: bool = False
    # List of initial validators.
    validators: List[AccountInfo]

    @field_validator('genesis_time', mode='before')
    def parse_genesis_time_to_datetime(cls, v):
        from datetime import datetime
        if v is None:
            return v
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            s = v
            if s.endswith('Z'):
                s = s[:-1] + '+00:00'
            try:
                return datetime.fromisoformat(s)
            except Exception as e:
                raise ValueError(f"genesis_time must be an ISO-8601 datetime string: {e}")
        raise TypeError('genesis_time must be a datetime or ISO-8601 string')

