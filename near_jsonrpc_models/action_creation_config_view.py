"""Describes the cost of creating a specific action, `Action`. Includes all variants."""

from near_jsonrpc_models.access_key_creation_config_view import AccessKeyCreationConfigView
from near_jsonrpc_models.fee import Fee
from pydantic import BaseModel


class ActionCreationConfigView(BaseModel):
    # Base cost of adding a key.
    add_key_cost: AccessKeyCreationConfigView = None
    # Base cost of creating an account.
    create_account_cost: Fee = None
    # Base cost for processing a delegate action.
    # 
    # This is on top of the costs for the actions inside the delegate action.
    delegate_cost: Fee = None
    # Base cost of deleting an account.
    delete_account_cost: Fee = None
    # Base cost of deleting a key.
    delete_key_cost: Fee = None
    # Base cost of deploying a contract.
    deploy_contract_cost: Fee = None
    # Cost per byte of deploying a contract.
    deploy_contract_cost_per_byte: Fee = None
    # Base cost of calling a function.
    function_call_cost: Fee = None
    # Cost per byte of method name and arguments of calling a function.
    function_call_cost_per_byte: Fee = None
    # Base cost of staking.
    stake_cost: Fee = None
    # Base cost of making a transfer.
    transfer_cost: Fee = None
