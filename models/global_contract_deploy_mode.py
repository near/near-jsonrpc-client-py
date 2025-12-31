"""CodeHash: Contract is deployed under its code hash.
Users will be able reference it by that hash.
This effectively makes the contract immutable.AccountId: Contract is deployed under the owner account id.
Users will be able reference it by that account id.
This allows the owner to update the contract for all its users."""

from pydantic import RootModel
from typing import Literal


class GlobalContractDeployMode(RootModel[Literal['CodeHash', 'AccountId']]):
    pass

