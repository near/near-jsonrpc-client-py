"""The initial state of a `0s` or `0u` account, as its state-init action carries it."""

from near_jsonrpc_models.global_contract_identifier_view import GlobalContractIdentifierView
from near_jsonrpc_models.raw_state_init import RawStateInit
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Dict
from typing import Union


class StateInitViewDeterministicPayload(BaseModel):
    code: GlobalContractIdentifierView
    data: Dict[str, str]

class StateInitViewDeterministic(StrictBaseModel):
    deterministic: StateInitViewDeterministicPayload

class StateInitViewUniversal(StrictBaseModel):
    universal: RawStateInit

class StateInitView(RootModel[Union[StateInitViewDeterministic, StateInitViewUniversal]]):
    pass

