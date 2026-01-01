"""Deploy global contract action"""

from near_jsonrpc_models.global_contract_deploy_mode import GlobalContractDeployMode
from pydantic import BaseModel


class DeployGlobalContractAction(BaseModel):
    # WebAssembly binary
    code: str
    deploy_mode: GlobalContractDeployMode
