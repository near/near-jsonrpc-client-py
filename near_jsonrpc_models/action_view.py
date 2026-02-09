from near_jsonrpc_models.access_key_view import AccessKeyView
from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.delegate_action import DelegateAction
from near_jsonrpc_models.function_args import FunctionArgs
from near_jsonrpc_models.global_contract_identifier_view import GlobalContractIdentifierView
from near_jsonrpc_models.near_gas import NearGas
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.signature import Signature
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Dict
from typing import Literal
from typing import Union


class ActionViewCreateAccount(RootModel[Literal['CreateAccount']]):
    pass

class ActionViewDeployContractPayload(BaseModel):
    code: str

class ActionViewDeployContract(StrictBaseModel):
    DeployContract: ActionViewDeployContractPayload

class ActionViewFunctionCallPayload(BaseModel):
    args: FunctionArgs
    deposit: NearToken
    gas: NearGas
    method_name: str

class ActionViewFunctionCall(StrictBaseModel):
    FunctionCall: ActionViewFunctionCallPayload

class ActionViewTransferPayload(BaseModel):
    deposit: NearToken

class ActionViewTransfer(StrictBaseModel):
    Transfer: ActionViewTransferPayload

class ActionViewStakePayload(BaseModel):
    public_key: PublicKey
    stake: NearToken

class ActionViewStake(StrictBaseModel):
    Stake: ActionViewStakePayload

class ActionViewAddKeyPayload(BaseModel):
    access_key: AccessKeyView
    public_key: PublicKey

class ActionViewAddKey(StrictBaseModel):
    AddKey: ActionViewAddKeyPayload

class ActionViewDeleteKeyPayload(BaseModel):
    public_key: PublicKey

class ActionViewDeleteKey(StrictBaseModel):
    DeleteKey: ActionViewDeleteKeyPayload

class ActionViewDeleteAccountPayload(BaseModel):
    beneficiary_id: AccountId

class ActionViewDeleteAccount(StrictBaseModel):
    DeleteAccount: ActionViewDeleteAccountPayload

class ActionViewDelegatePayload(BaseModel):
    delegate_action: DelegateAction
    signature: Signature

class ActionViewDelegate(StrictBaseModel):
    Delegate: ActionViewDelegatePayload

class ActionViewDeployGlobalContractPayload(BaseModel):
    code: str

class ActionViewDeployGlobalContract(StrictBaseModel):
    DeployGlobalContract: ActionViewDeployGlobalContractPayload

class ActionViewDeployGlobalContractByAccountIdPayload(BaseModel):
    code: str

class ActionViewDeployGlobalContractByAccountId(StrictBaseModel):
    DeployGlobalContractByAccountId: ActionViewDeployGlobalContractByAccountIdPayload

class ActionViewUseGlobalContractPayload(BaseModel):
    code_hash: CryptoHash

class ActionViewUseGlobalContract(StrictBaseModel):
    UseGlobalContract: ActionViewUseGlobalContractPayload

class ActionViewUseGlobalContractByAccountIdPayload(BaseModel):
    account_id: AccountId

class ActionViewUseGlobalContractByAccountId(StrictBaseModel):
    UseGlobalContractByAccountId: ActionViewUseGlobalContractByAccountIdPayload

class ActionViewDeterministicStateInitPayload(BaseModel):
    code: GlobalContractIdentifierView
    data: Dict[str, str]
    deposit: NearToken

class ActionViewDeterministicStateInit(StrictBaseModel):
    DeterministicStateInit: ActionViewDeterministicStateInitPayload

class ActionViewTransferToGasKeyPayload(BaseModel):
    deposit: NearToken
    public_key: PublicKey

class ActionViewTransferToGasKey(StrictBaseModel):
    TransferToGasKey: ActionViewTransferToGasKeyPayload

class ActionViewWithdrawFromGasKeyPayload(BaseModel):
    amount: NearToken
    public_key: PublicKey

class ActionViewWithdrawFromGasKey(StrictBaseModel):
    WithdrawFromGasKey: ActionViewWithdrawFromGasKeyPayload

class ActionView(RootModel[Union[ActionViewCreateAccount, ActionViewDeployContract, ActionViewFunctionCall, ActionViewTransfer, ActionViewStake, ActionViewAddKey, ActionViewDeleteKey, ActionViewDeleteAccount, ActionViewDelegate, ActionViewDeployGlobalContract, ActionViewDeployGlobalContractByAccountId, ActionViewUseGlobalContract, ActionViewUseGlobalContractByAccountId, ActionViewDeterministicStateInit, ActionViewTransferToGasKey, ActionViewWithdrawFromGasKey]]):
    pass

