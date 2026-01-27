from near_jsonrpc_models.deterministic_account_state_init import DeterministicAccountStateInit
from near_jsonrpc_models.near_token import NearToken
from pydantic import BaseModel


class DeterministicStateInitAction(BaseModel):
    deposit: NearToken
    state_init: DeterministicAccountStateInit
