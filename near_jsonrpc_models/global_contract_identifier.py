from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class GlobalContractIdentifierHash(StrictBaseModel):
    hash: CryptoHash

class GlobalContractIdentifierAccountId(StrictBaseModel):
    account_id: AccountId

class GlobalContractIdentifier(RootModel[Union[GlobalContractIdentifierHash, GlobalContractIdentifierAccountId]]):
    pass

