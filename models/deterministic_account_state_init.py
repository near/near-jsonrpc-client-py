from models.deterministic_account_state_init_v1 import DeterministicAccountStateInitV1
from models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class DeterministicAccountStateInitV1Option(StrictBaseModel):
    V1: DeterministicAccountStateInitV1

class DeterministicAccountStateInit(RootModel[Union[DeterministicAccountStateInitV1Option]]):
    pass

