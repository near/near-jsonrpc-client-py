"""View that preserves JSON format of the runtime config."""

from models.account_creation_config_view import AccountCreationConfigView
from models.congestion_control_config_view import CongestionControlConfigView
from models.dynamic_resharding_config_view import DynamicReshardingConfigView
from models.near_token import NearToken
from models.runtime_fees_config_view import RuntimeFeesConfigView
from models.vmconfig_view import VMConfigView
from models.witness_config_view import WitnessConfigView
from pydantic import BaseModel
from pydantic import Field


class RuntimeConfigView(BaseModel):
    # Config that defines rules for account creation.
    account_creation_config: AccountCreationConfigView = None
    # The configuration for congestion control.
    congestion_control_config: CongestionControlConfigView = None
    # Configuration for dynamic resharding feature.
    dynamic_resharding_config: DynamicReshardingConfigView = Field(default_factory=lambda: DynamicReshardingConfigView(**{'max_number_of_shards': 999999999999999, 'memory_usage_threshold': 999999999999999, 'min_child_memory_usage': 999999999999999, 'min_epochs_between_resharding': 999999999999999}))
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
