from near_jsonrpc_models.ext_costs_config_view import ExtCostsConfigView
from near_jsonrpc_models.limit_config import LimitConfig
from near_jsonrpc_models.vmkind import VMKind
from pydantic import BaseModel
from pydantic import conint


class VMConfigView(BaseModel):
    # See [VMConfig::bls12381_not_in_group_fix](crate::vm::Config::bls12381_not_in_group_fix).
    bls12381_not_in_group_fix: bool = None
    # Costs for runtime externals
    ext_costs: ExtCostsConfigView = None
    # See [VMConfig::fix_contract_loading_cost](crate::vm::Config::fix_contract_loading_cost).
    fix_contract_loading_cost: bool = None
    # Gas cost of a growing memory by single page.
    grow_mem_cost: conint(ge=0, le=4294967295) = None
    # Describes limits for VM and Runtime.
    # 
    # TODO: Consider changing this to `VMLimitConfigView` to avoid dependency
    # on runtime.
    limit_config: LimitConfig = None
    # Base gas cost of a linear operation
    linear_op_base_cost: conint(ge=0, le=18446744073709551615) = None
    # Unit gas cost of a linear operation
    linear_op_unit_cost: conint(ge=0, le=18446744073709551615) = None
    # See [VMConfig::ml_dsa_verify_host_fn](crate::vm::Config::ml_dsa_verify_host_fn).
    ml_dsa_verify_host_fn: bool = None
    # Gas cost of a regular operation.
    regular_op_cost: conint(ge=0, le=4294967295) = None
    # See [VMConfig::sha3_host_fns](crate::vm::Config::sha3_host_fns).
    sha3_host_fns: bool = None
    # See [VMConfig::universal_accounts](crate::vm::Config::universal_accounts).
    universal_accounts: bool = None
    # See [VMConfig::vm_kind](crate::vm::Config::vm_kind).
    vm_kind: VMKind = None
