from near_jsonrpc_models.deterministic_account_state_init_v1 import DeterministicAccountStateInitV1
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class DeterministicAccountStateInitV2(StrictBaseModel):
    V1: DeterministicAccountStateInitV1

class DeterministicAccountStateInit(RootModel[Union[DeterministicAccountStateInitV2]]):
    pass

