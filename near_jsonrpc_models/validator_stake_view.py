from near_jsonrpc_models.validator_stake_view_v1 import ValidatorStakeViewV1
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ValidatorStakeViewValidatorStakeStructVersion(ValidatorStakeViewV1):
    validator_stake_struct_version: Literal['V1']

class ValidatorStakeView(RootModel[Union[ValidatorStakeViewValidatorStakeStructVersion]]):
    pass

