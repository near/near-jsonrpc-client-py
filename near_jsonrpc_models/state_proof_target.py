"""Which piece of a shard's state a light-client state proof targets.

An account that runs a global contract has no local code, so `LocalContractCode` is
absent for it. `Account::contract()` says which case applies."""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.store_key import StoreKey
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class StateProofTargetAccountIdTargetType(BaseModel):
    account_id: AccountId
    target_type: Literal['account']

class StateProofTargetAccountIdTargetType1(BaseModel):
    account_id: AccountId
    target_type: Literal['local_contract_code']

class StateProofTargetAccountIdKeyTargetType(BaseModel):
    account_id: AccountId
    key: StoreKey
    target_type: Literal['contract_data']

class StateProofTargetAccountIdPublicKeyTargetType(BaseModel):
    account_id: AccountId
    public_key: PublicKey
    target_type: Literal['access_key']

class StateProofTarget(RootModel[Union[StateProofTargetAccountIdTargetType, StateProofTargetAccountIdTargetType1, StateProofTargetAccountIdKeyTargetType, StateProofTargetAccountIdPublicKeyTargetType]]):
    pass

