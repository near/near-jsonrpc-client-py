from datetime import datetime
from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.near_gas import NearGas
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.runtime_config_view import RuntimeConfigView
from near_jsonrpc_models.shard_layout import ShardLayout
from pydantic import BaseModel
from pydantic import conint
from pydantic import conlist
from pydantic import field_validator
from typing import List


class RpcProtocolConfigResponse(BaseModel):
    # Expected number of hidden validators per shard.
    avg_hidden_validator_seats_per_shard: List[conint(ge=0, le=18446744073709551615)] = None
    # Threshold for kicking out block producers, between 0 and 100.
    block_producer_kickout_threshold: conint(ge=0, le=255) = None
    # ID of the blockchain. This must be unique for every blockchain.
    # If your testnet blockchains do not have unique chain IDs, you will have a bad time.
    chain_id: str = None
    # Threshold for kicking out chunk producers, between 0 and 100.
    chunk_producer_kickout_threshold: conint(ge=0, le=255) = None
    # Threshold for kicking out nodes which are only chunk validators, between 0 and 100.
    chunk_validator_only_kickout_threshold: conint(ge=0, le=255) = None
    # Enable dynamic re-sharding.
    dynamic_resharding: bool = None
    # Epoch length counted in block heights.
    epoch_length: conint(ge=0, le=18446744073709551615) = None
    # Fishermen stake threshold.
    fishermen_threshold: NearToken = None
    # Initial gas limit.
    gas_limit: NearGas = None
    # Gas price adjustment rate
    gas_price_adjustment_rate: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = None
    # Height of genesis block.
    genesis_height: conint(ge=0, le=18446744073709551615) = None
    # Official time of blockchain start.
    genesis_time: datetime = None
    # Maximum gas price.
    max_gas_price: NearToken = None
    # Maximum inflation on the total supply every epoch.
    max_inflation_rate: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = None
    # Max stake percentage of the validators we will kick out.
    max_kickout_stake_perc: conint(ge=0, le=255) = None
    # Minimum gas price. It is also the initial gas price.
    min_gas_price: NearToken = None
    # The minimum stake required for staking is last seat price divided by this number.
    minimum_stake_divisor: conint(ge=0, le=18446744073709551615) = None
    # The lowest ratio s/s_total any block producer can have.
    # See <https://github.com/near/NEPs/pull/167> for details
    minimum_stake_ratio: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = None
    # The minimum number of validators each shard must have
    minimum_validators_per_shard: conint(ge=0, le=18446744073709551615) = None
    # Number of block producer seats at genesis.
    num_block_producer_seats: conint(ge=0, le=18446744073709551615) = None
    # Defines number of shards and number of block producer seats per each shard at genesis.
    num_block_producer_seats_per_shard: List[conint(ge=0, le=18446744073709551615)] = None
    # Expected number of blocks per year
    num_blocks_per_year: conint(ge=0, le=18446744073709551615) = None
    # Online maximum threshold above which validator gets full reward.
    online_max_threshold: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = None
    # Online minimum threshold below which validator doesn't receive reward.
    online_min_threshold: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = None
    # Protocol treasury rate
    protocol_reward_rate: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = None
    # Protocol treasury account
    protocol_treasury_account: AccountId = None
    # Threshold of stake that needs to indicate that they ready for upgrade.
    protocol_upgrade_stake_threshold: conlist(conint(ge=-2147483648, le=2147483647), min_length=2, max_length=2) = None
    # Current Protocol Version
    protocol_version: conint(ge=0, le=4294967295) = None
    # Runtime configuration (mostly economics constants).
    runtime_config: RuntimeConfigView = None
    # Layout information regarding how to split accounts to shards
    shard_layout: ShardLayout = None
    # If true, shuffle the chunk producers across shards. In other words, if
    # the shard assignments were `[S_0, S_1, S_2, S_3]` where `S_i` represents
    # the set of chunk producers for shard `i`, if this flag were true, the
    # shard assignments might become, for example, `[S_2, S_0, S_3, S_1]`.
    shuffle_shard_assignment_for_chunk_producers: bool = None
    # Number of target chunk validator mandates for each shard.
    target_validator_mandates_per_shard: conint(ge=0, le=18446744073709551615) = None
    # Number of blocks for which a given transaction is valid
    transaction_validity_period: conint(ge=0, le=18446744073709551615) = None

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

