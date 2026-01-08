"""Describes limits for VM and Runtime.
TODO #4139: consider switching to strongly-typed wrappers instead of raw quantities"""

from near_jsonrpc_models.account_id_validity_rules_version import AccountIdValidityRulesVersion
from near_jsonrpc_models.near_gas import NearGas
from pydantic import BaseModel
from pydantic import Field
from pydantic import conint


class LimitConfig(BaseModel):
    # Whether to enforce account_id well-formed-ness where it wasn't enforced
    # historically.
    account_id_validity_rules_version: AccountIdValidityRulesVersion = Field(default_factory=lambda: AccountIdValidityRulesVersion(0))
    # The initial number of memory pages.
    # NOTE: It's not a limiter itself, but it's a value we use for initial_memory_pages.
    initial_memory_pages: conint(ge=0, le=4294967295) = None
    # Max number of actions per receipt.
    max_actions_per_receipt: conint(ge=0, le=18446744073709551615) = None
    # Max length of arguments in a function call action.
    max_arguments_length: conint(ge=0, le=18446744073709551615) = None
    # Max contract size
    max_contract_size: conint(ge=0, le=18446744073709551615) = None
    # If present, stores max number of elements in a single contract's table
    max_elements_per_contract_table: conint(ge=0, le=4294967295) | None = None
    # If present, stores max number of functions in one contract
    max_functions_number_per_contract: conint(ge=0, le=18446744073709551615) | None = None
    # Max amount of gas that can be used, excluding gas attached to promises.
    max_gas_burnt: NearGas = None
    # Max length of any method name (without terminating character).
    max_length_method_name: conint(ge=0, le=18446744073709551615) = None
    # Max length of returned data
    max_length_returned_data: conint(ge=0, le=18446744073709551615) = None
    # Max storage key size
    max_length_storage_key: conint(ge=0, le=18446744073709551615) = None
    # Max storage value size
    max_length_storage_value: conint(ge=0, le=18446744073709551615) = None
    # If present, stores max number of locals declared globally in one contract
    max_locals_per_contract: conint(ge=0, le=18446744073709551615) | None = None
    # What is the maximal memory pages amount is allowed to have for a contract.
    max_memory_pages: conint(ge=0, le=4294967295) = None
    # Max total length of all method names (including terminating character) for a function call
    # permission access key.
    max_number_bytes_method_names: conint(ge=0, le=18446744073709551615) = None
    # Max number of input data dependencies
    max_number_input_data_dependencies: conint(ge=0, le=18446744073709551615) = None
    # Maximum number of log entries.
    max_number_logs: conint(ge=0, le=18446744073709551615) = None
    # Maximum number of registers that can be used simultaneously.
    # 
    # Note that due to an implementation quirk [read: a bug] in VMLogic, if we
    # have this number of registers, no subsequent writes to the registers
    # will succeed even if they replace an existing register.
    max_number_registers: conint(ge=0, le=18446744073709551615) = None
    # Max number of promises that a function call can create
    max_promises_per_function_call_action: conint(ge=0, le=18446744073709551615) = None
    # Max receipt size
    max_receipt_size: conint(ge=0, le=18446744073709551615) = None
    # Maximum number of bytes that can be stored in a single register.
    max_register_size: conint(ge=0, le=18446744073709551615) = None
    # How tall the stack is allowed to grow?
    # 
    # See <https://wiki.parity.io/WebAssembly-StackHeight> to find out how the stack frame cost
    # is calculated.
    max_stack_height: conint(ge=0, le=4294967295) = None
    # If present, stores max number of tables declared globally in one contract
    max_tables_per_contract: conint(ge=0, le=4294967295) | None = None
    # Maximum total length in bytes of all log messages.
    max_total_log_length: conint(ge=0, le=18446744073709551615) = None
    # Max total prepaid gas for all function call actions per receipt.
    max_total_prepaid_gas: NearGas = None
    # Max transaction size
    max_transaction_size: conint(ge=0, le=18446744073709551615) = None
    # Maximum number of bytes for payload passed over a yield resume.
    max_yield_payload_size: conint(ge=0, le=18446744073709551615) = None
    # Hard limit on the size of storage proof generated while executing a single receipt.
    per_receipt_storage_proof_size_limit: conint(ge=0, le=4294967295) = None
    # Limit of memory used by registers.
    registers_memory_limit: conint(ge=0, le=18446744073709551615) = None
    # Number of blocks after which a yielded promise times out.
    yield_timeout_length_in_blocks: conint(ge=0, le=18446744073709551615) = None
