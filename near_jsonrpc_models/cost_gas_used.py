"""Shows gas profile. More info [here](https://near.github.io/nearcore/architecture/gas/gas_profile.html?highlight=WASM_HOST_COST#example-transaction-gas-profile)."""

from pydantic import BaseModel


class CostGasUsed(BaseModel):
    cost: str
    # Either ACTION_COST or WASM_HOST_COST.
    cost_category: str
    gas_used: str
