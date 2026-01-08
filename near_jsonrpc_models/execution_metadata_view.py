from near_jsonrpc_models.cost_gas_used import CostGasUsed
from pydantic import BaseModel
from pydantic import conint
from typing import List


class ExecutionMetadataView(BaseModel):
    gas_profile: List[CostGasUsed] | None = None
    version: conint(ge=0, le=4294967295)
