"""It is a [serializable view] of [`StateChangeKind`].

[serializable view]: ./index.html
[`StateChangeKind`]: ../types/struct.StateChangeKind.html"""

from near_jsonrpc_models.account_id import AccountId
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class StateChangeKindViewAccountTouched(BaseModel):
    account_id: AccountId
    type: Literal['account_touched']

class StateChangeKindViewAccessKeyTouched(BaseModel):
    account_id: AccountId
    type: Literal['access_key_touched']

class StateChangeKindViewDataTouched(BaseModel):
    account_id: AccountId
    type: Literal['data_touched']

class StateChangeKindViewContractCodeTouched(BaseModel):
    account_id: AccountId
    type: Literal['contract_code_touched']

class StateChangeKindView(RootModel[Union[StateChangeKindViewAccountTouched, StateChangeKindViewAccessKeyTouched, StateChangeKindViewDataTouched, StateChangeKindViewContractCodeTouched]]):
    pass

