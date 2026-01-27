"""Use global contract action"""

from near_jsonrpc_models.global_contract_identifier import GlobalContractIdentifier
from pydantic import BaseModel


class UseGlobalContractAction(BaseModel):
    contract_identifier: GlobalContractIdentifier
