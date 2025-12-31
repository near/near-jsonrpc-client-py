from models.account_id import AccountId
from models.near_token import NearToken
from models.public_key import PublicKey
from models.shard_id import ShardId
from pydantic import BaseModel
from typing import List


class NextEpochValidatorInfo(BaseModel):
    account_id: AccountId
    public_key: PublicKey
    shards: List[ShardId]
    stake: NearToken
