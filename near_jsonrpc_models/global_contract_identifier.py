from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class GlobalContractIdentifierCodeHash(StrictBaseModel):
    CodeHash: CryptoHash

class GlobalContractIdentifierAccountId(StrictBaseModel):
    AccountId: AccountId

class GlobalContractIdentifier(RootModel[Union[GlobalContractIdentifierCodeHash, GlobalContractIdentifierAccountId]]):
    pass

