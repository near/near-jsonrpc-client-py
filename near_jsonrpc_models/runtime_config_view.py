"""View that preserves JSON format of the runtime config."""

from near_jsonrpc_models.account_creation_config_view import AccountCreationConfigView
from near_jsonrpc_models.congestion_control_config_view import CongestionControlConfigView
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.runtime_fees_config_view import RuntimeFeesConfigView
from near_jsonrpc_models.vmconfig_view import VMConfigView
from near_jsonrpc_models.witness_config_view import WitnessConfigView
from pydantic import BaseModel


class RuntimeConfigView(BaseModel):
    # Config that defines rules for account creation.
    account_creation_config: AccountCreationConfigView = None
    # The configuration for congestion control.
    congestion_control_config: CongestionControlConfigView = None
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
