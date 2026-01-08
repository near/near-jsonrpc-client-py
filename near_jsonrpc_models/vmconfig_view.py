from near_jsonrpc_models.ext_costs_config_view import ExtCostsConfigView
from near_jsonrpc_models.limit_config import LimitConfig
from near_jsonrpc_models.storage_get_mode import StorageGetMode
from near_jsonrpc_models.vmkind import VMKind
from pydantic import BaseModel
from pydantic import conint


class VMConfigView(BaseModel):
    # See [VMConfig::deterministic_account_ids](crate::vm::Config::deterministic_account_ids).
    deterministic_account_ids: bool = None
    # See [VMConfig::discard_custom_sections](crate::vm::Config::discard_custom_sections).
    discard_custom_sections: bool = None
    # See [VMConfig::eth_implicit_accounts](crate::vm::Config::eth_implicit_accounts).
    eth_implicit_accounts: bool = None
    # Costs for runtime externals
    ext_costs: ExtCostsConfigView = None
    # See [VMConfig::fix_contract_loading_cost](crate::vm::Config::fix_contract_loading_cost).
    fix_contract_loading_cost: bool = None
    # See [VMConfig::global_contract_host_fns](crate::vm::Config::global_contract_host_fns).
    global_contract_host_fns: bool = None
    # Gas cost of a growing memory by single page.
    grow_mem_cost: conint(ge=0, le=4294967295) = None
    # Deprecated
    implicit_account_creation: bool = None
    # Describes limits for VM and Runtime.
    # 
    # TODO: Consider changing this to `VMLimitConfigView` to avoid dependency
    # on runtime.
    limit_config: LimitConfig = None
    # Base gas cost of a linear operation
    linear_op_base_cost: conint(ge=0, le=18446744073709551615) = None
    # Unit gas cost of a linear operation
    linear_op_unit_cost: conint(ge=0, le=18446744073709551615) = None
    # See [VMConfig::reftypes_bulk_memory](crate::vm::Config::reftypes_bulk_memory).
    reftypes_bulk_memory: bool = None
    # Gas cost of a regular operation.
    regular_op_cost: conint(ge=0, le=4294967295) = None
    # See [VMConfig::storage_get_mode](crate::vm::Config::storage_get_mode).
    storage_get_mode: StorageGetMode = None
    # See [VMConfig::vm_kind](crate::vm::Config::vm_kind).
    vm_kind: VMKind = None
