from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel
from typing import List


class NextEpochValidatorInfo(BaseModel):
    account_id: AccountId
    public_key: PublicKey
    shards: List[ShardId]
    stake: NearToken
