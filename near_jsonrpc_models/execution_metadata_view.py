from near_jsonrpc_models.account_contract_view import AccountContractView
from near_jsonrpc_models.cost_gas_used import CostGasUsed
from pydantic import BaseModel
from pydantic import conint
from typing import List


class ExecutionMetadataView(BaseModel):
    # One entry per action in the receipt (V4+ only): the contract attached
    # to the receiver account immediately before that action ran. The inner
    # `Option` is `Some` (a tagged contract object) when the account had a
    # contract and `None` (rendered as JSON `null`) when it did not (e.g. an
    # account with no code, or one that did not yet exist). The outer
    # `Option` is `None` for older metadata versions.
    contracts: List[AccountContractView | None] | None = None
    gas_profile: List[CostGasUsed] | None = None
    version: conint(ge=0, le=4294967295)
