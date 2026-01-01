from near_jsonrpc_models.global_contract_identifier import GlobalContractIdentifier
from pydantic import BaseModel
from typing import Dict


class DeterministicAccountStateInitV1(BaseModel):
    code: GlobalContractIdentifier
    data: Dict[str, str]
