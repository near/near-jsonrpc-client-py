"""An Action that can be included in a transaction or receipt, excluding delegate actions. This type represents all possible action types except DelegateAction to prevent infinite recursion in meta-transactions."""

from near_jsonrpc_models.add_key_action import AddKeyAction
from near_jsonrpc_models.create_account_action import CreateAccountAction
from near_jsonrpc_models.delete_account_action import DeleteAccountAction
from near_jsonrpc_models.delete_key_action import DeleteKeyAction
from near_jsonrpc_models.deploy_contract_action import DeployContractAction
from near_jsonrpc_models.deploy_global_contract_action import DeployGlobalContractAction
from near_jsonrpc_models.deterministic_state_init_action import DeterministicStateInitAction
from near_jsonrpc_models.function_call_action import FunctionCallAction
from near_jsonrpc_models.stake_action import StakeAction
from near_jsonrpc_models.strict_model import StrictBaseModel
from near_jsonrpc_models.transfer_action import TransferAction
from near_jsonrpc_models.transfer_to_gas_key_action import TransferToGasKeyAction
from near_jsonrpc_models.use_global_contract_action import UseGlobalContractAction
from near_jsonrpc_models.withdraw_from_gas_key_action import WithdrawFromGasKeyAction
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class NonDelegateActionCreateAccount(StrictBaseModel):
    """Create an (sub)account using a transaction `receiver_id` as an ID for
a new account ID must pass validation rules described here
<https://nomicon.io/DataStructures/Account>."""
    CreateAccount: CreateAccountAction

class NonDelegateActionDeployContract(StrictBaseModel):
    """Sets a Wasm code to a receiver_id"""
    DeployContract: DeployContractAction

class NonDelegateActionFunctionCall(StrictBaseModel):
    FunctionCall: FunctionCallAction

class NonDelegateActionTransfer(StrictBaseModel):
    Transfer: TransferAction

class NonDelegateActionStake(StrictBaseModel):
    Stake: StakeAction

class NonDelegateActionAddKey(StrictBaseModel):
    AddKey: AddKeyAction

class NonDelegateActionDeleteKey(StrictBaseModel):
    DeleteKey: DeleteKeyAction

class NonDelegateActionDeleteAccount(StrictBaseModel):
    DeleteAccount: DeleteAccountAction

class NonDelegateActionDeployGlobalContract(StrictBaseModel):
    DeployGlobalContract: DeployGlobalContractAction

class NonDelegateActionUseGlobalContract(StrictBaseModel):
    UseGlobalContract: UseGlobalContractAction

class NonDelegateActionDeterministicStateInit(StrictBaseModel):
    DeterministicStateInit: DeterministicStateInitAction

class NonDelegateActionTransferToGasKey(StrictBaseModel):
    TransferToGasKey: TransferToGasKeyAction

class NonDelegateActionWithdrawFromGasKey(StrictBaseModel):
    WithdrawFromGasKey: WithdrawFromGasKeyAction

class NonDelegateAction(RootModel[Union[NonDelegateActionCreateAccount, NonDelegateActionDeployContract, NonDelegateActionFunctionCall, NonDelegateActionTransfer, NonDelegateActionStake, NonDelegateActionAddKey, NonDelegateActionDeleteKey, NonDelegateActionDeleteAccount, NonDelegateActionDeployGlobalContract, NonDelegateActionUseGlobalContract, NonDelegateActionDeterministicStateInit, NonDelegateActionTransferToGasKey, NonDelegateActionWithdrawFromGasKey]]):
    pass

