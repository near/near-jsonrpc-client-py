from models.access_key_view import AccessKeyView
from models.account_id import AccountId
from models.crypto_hash import CryptoHash
from models.gas_key import GasKey
from models.near_token import NearToken
from models.public_key import PublicKey
from models.state_change_cause_view import StateChangeCauseView
from models.store_key import StoreKey
from models.store_value import StoreValue
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


class StateChangeWithCauseViewChangePayload(BaseModel):
    account_id: AccountId
    amount: NearToken
    code_hash: CryptoHash
    global_contract_account_id: AccountId | None = None
    global_contract_hash: CryptoHash | None = None
    locked: NearToken
    # TODO(2271): deprecated.
    storage_paid_at: conint(ge=0, le=18446744073709551615) = 0
    storage_usage: conint(ge=0, le=18446744073709551615)

class StateChangeWithCauseViewChange(BaseModel):
    cause: StateChangeCauseView
    # A view of the account
    change: StateChangeWithCauseViewChangePayload
    type: Literal['account_update']

class StateChangeWithCauseViewChangeOptionChange(BaseModel):
    account_id: AccountId

class StateChangeWithCauseViewChangeOption(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewChangeOptionChange
    type: Literal['account_deletion']

class StateChangeWithCauseViewChange1Change(BaseModel):
    access_key: AccessKeyView
    account_id: AccountId
    public_key: PublicKey

class StateChangeWithCauseViewChange1(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewChange1Change
    type: Literal['access_key_update']

class StateChangeWithCauseViewChange2Change(BaseModel):
    account_id: AccountId
    public_key: PublicKey

class StateChangeWithCauseViewChange2(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewChange2Change
    type: Literal['access_key_deletion']

class StateChangeWithCauseViewChange3Change(BaseModel):
    account_id: AccountId
    gas_key: GasKey
    public_key: PublicKey

class StateChangeWithCauseViewChange3(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewChange3Change
    type: Literal['gas_key_update']

class StateChangeWithCauseViewChange4Change(BaseModel):
    account_id: AccountId
    index: conint(ge=0, le=4294967295)
    nonce: conint(ge=0, le=18446744073709551615)
    public_key: PublicKey

class StateChangeWithCauseViewChange4(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewChange4Change
    type: Literal['gas_key_nonce_update']

class StateChangeWithCauseViewChange5Change(BaseModel):
    account_id: AccountId
    public_key: PublicKey

class StateChangeWithCauseViewChange5(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewChange5Change
    type: Literal['gas_key_deletion']

class StateChangeWithCauseViewChange6Change(BaseModel):
    account_id: AccountId
    key_base64: StoreKey
    value_base64: StoreValue

class StateChangeWithCauseViewChange6(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewChange6Change
    type: Literal['data_update']

class StateChangeWithCauseViewChange7Change(BaseModel):
    account_id: AccountId
    key_base64: StoreKey

class StateChangeWithCauseViewChange7(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewChange7Change
    type: Literal['data_deletion']

class StateChangeWithCauseViewChange8Change(BaseModel):
    account_id: AccountId
    code_base64: str

class StateChangeWithCauseViewChange8(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewChange8Change
    type: Literal['contract_code_update']

class StateChangeWithCauseViewChange9Change(BaseModel):
    account_id: AccountId

class StateChangeWithCauseViewChange9(BaseModel):
    cause: StateChangeCauseView
    change: StateChangeWithCauseViewChange9Change
    type: Literal['contract_code_deletion']

class StateChangeWithCauseView(RootModel[Union[StateChangeWithCauseViewChange, StateChangeWithCauseViewChangeOption, StateChangeWithCauseViewChange1, StateChangeWithCauseViewChange2, StateChangeWithCauseViewChange3, StateChangeWithCauseViewChange4, StateChangeWithCauseViewChange5, StateChangeWithCauseViewChange6, StateChangeWithCauseViewChange7, StateChangeWithCauseViewChange8, StateChangeWithCauseViewChange9]]):
    pass

