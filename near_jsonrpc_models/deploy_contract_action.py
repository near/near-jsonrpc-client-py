"""Deploy contract action"""

from pydantic import BaseModel


class DeployContractAction(BaseModel):
    # WebAssembly binary
    code: str
