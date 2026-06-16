"""View that preserves JSON format of the runtime config."""

from near_jsonrpc_models.account_creation_config_view import AccountCreationConfigView
from near_jsonrpc_models.congestion_control_config_view import CongestionControlConfigView
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.runtime_fees_config_view import RuntimeFeesConfigView
from near_jsonrpc_models.vmconfig_view import VMConfigView
from near_jsonrpc_models.witness_config_view import WitnessConfigView
from pydantic import BaseModel
from pydantic import Field


class RuntimeConfigView(BaseModel):
    # How much creating an account should cost in NEAR. Taken into account when burning gas for
    # account creation.
    account_creation_charge: NearToken = Field(default_factory=lambda: NearToken('0'))
    # Config that defines rules for account creation.
    account_creation_config: AccountCreationConfigView = None
    # The configuration for congestion control.
    congestion_control_config: CongestionControlConfigView = None
    # Minimum price at which the gas attached to a receipt is purchased. The price at which it is
    # burned might be lower, in which case the difference is refunded after execution.
    min_gas_purchase_price: NearToken = Field(default_factory=lambda: NearToken('0'))
    # Amount of yN per byte required to have on the account.  See
    # <https://nomicon.io/Economics/Economics.html#state-stake> for details.
    storage_amount_per_byte: NearToken = None
    # Costs of different actions that need to be performed when sending and
    # processing transaction and receipts.
    transaction_costs: RuntimeFeesConfigView = None
    # Config of wasm operations.
    wasm_config: VMConfigView = None
    # Configuration specific to ChunkStateWitness.
    witness_config: WitnessConfigView = None
