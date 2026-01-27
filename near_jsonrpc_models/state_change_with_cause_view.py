from near_jsonrpc_models.access_key_view import AccessKeyView
from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.state_change_cause_view import StateChangeCauseView
from near_jsonrpc_models.store_key import StoreKey
from near_jsonrpc_models.store_value import StoreValue
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class StateChangeWithCauseViewAccountUpdateChange(BaseModel):
    account_id: AccountId
    amount: NearToken
    code_hash: CryptoHash
    global_contract_account_id: AccountId | None = None
    global_contract_hash: CryptoHash | None = None
    locked: NearToken
    # TODO(2271): deprecated.
    storage_paid_at: conint(ge=0, le=18446744073709551615) = 0
    storage_usage: conint(ge=0, le=18446744073709551615)

class StateChangeWithCauseViewAccountUpdate(BaseModel):
    cause: StateChangeCauseView
    # A view of the account
    change: StateChangeWithCauseViewAccountUpdateChange
    type: Literal['account_update']

class StateChangeWithCauseViewAccountDeletionChange(BaseModel):
    account_id: AccountId

class StateChangeWithCauseViewAccountDeletion(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewAccountDeletionChange
    type: Literal['account_deletion']

class StateChangeWithCauseViewAccessKeyUpdateChange(BaseModel):
    access_key: AccessKeyView
    account_id: AccountId
    public_key: PublicKey

class StateChangeWithCauseViewAccessKeyUpdate(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewAccessKeyUpdateChange
    type: Literal['access_key_update']

class StateChangeWithCauseViewAccessKeyDeletionChange(BaseModel):
    account_id: AccountId
    public_key: PublicKey

class StateChangeWithCauseViewAccessKeyDeletion(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewAccessKeyDeletionChange
    type: Literal['access_key_deletion']

class StateChangeWithCauseViewGasKeyNonceUpdateChange(BaseModel):
    account_id: AccountId
    index: conint(ge=0, le=4294967295)
    nonce: conint(ge=0, le=18446744073709551615)
    public_key: PublicKey

class StateChangeWithCauseViewGasKeyNonceUpdate(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewGasKeyNonceUpdateChange
    type: Literal['gas_key_nonce_update']

class StateChangeWithCauseViewDataUpdateChange(BaseModel):
    account_id: AccountId
    key_base64: StoreKey
    value_base64: StoreValue

class StateChangeWithCauseViewDataUpdate(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewDataUpdateChange
    type: Literal['data_update']

class StateChangeWithCauseViewDataDeletionChange(BaseModel):
    account_id: AccountId
    key_base64: StoreKey

class StateChangeWithCauseViewDataDeletion(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewDataDeletionChange
    type: Literal['data_deletion']

class StateChangeWithCauseViewContractCodeUpdateChange(BaseModel):
    account_id: AccountId
    code_base64: str

class StateChangeWithCauseViewContractCodeUpdate(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewContractCodeUpdateChange
    type: Literal['contract_code_update']

class StateChangeWithCauseViewContractCodeDeletionChange(BaseModel):
    account_id: AccountId

class StateChangeWithCauseViewContractCodeDeletion(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewContractCodeDeletionChange
    type: Literal['contract_code_deletion']

class StateChangeWithCauseView(RootModel[Union[StateChangeWithCauseViewAccountUpdate, StateChangeWithCauseViewAccountDeletion, StateChangeWithCauseViewAccessKeyUpdate, StateChangeWithCauseViewAccessKeyDeletion, StateChangeWithCauseViewGasKeyNonceUpdate, StateChangeWithCauseViewDataUpdate, StateChangeWithCauseViewDataDeletion, StateChangeWithCauseViewContractCodeUpdate, StateChangeWithCauseViewContractCodeDeletion]]):
    pass

