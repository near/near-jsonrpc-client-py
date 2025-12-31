"""See crate::types::StateChangeCause for details."""

from models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class StateChangeCauseViewType(BaseModel):
    type: Literal['not_writable_to_disk']

class StateChangeCauseViewTypeOption(BaseModel):
    type: Literal['initial_state']

class StateChangeCauseViewTxHash(BaseModel):
    tx_hash: CryptoHash
    type: Literal['transaction_processing']

class StateChangeCauseViewReceiptHash(BaseModel):
    receipt_hash: CryptoHash
    type: Literal['action_receipt_processing_started']

class StateChangeCauseViewReceiptHashOption(BaseModel):
    receipt_hash: CryptoHash
    type: Literal['action_receipt_gas_reward']

class StateChangeCauseViewReceiptHash1(BaseModel):
    receipt_hash: CryptoHash
    type: Literal['receipt_processing']

class StateChangeCauseViewReceiptHash2(BaseModel):
    receipt_hash: CryptoHash
    type: Literal['postponed_receipt']

class StateChangeCauseViewType1(BaseModel):
    type: Literal['updated_delayed_receipts']

class StateChangeCauseViewType2(BaseModel):
    type: Literal['validator_accounts_update']

class StateChangeCauseViewType3(BaseModel):
    type: Literal['migration']

class StateChangeCauseViewType4(BaseModel):
    type: Literal['bandwidth_scheduler_state_update']

class StateChangeCauseView(RootModel[Union[StateChangeCauseViewType, StateChangeCauseViewTypeOption, StateChangeCauseViewTxHash, StateChangeCauseViewReceiptHash, StateChangeCauseViewReceiptHashOption, StateChangeCauseViewReceiptHash1, StateChangeCauseViewReceiptHash2, StateChangeCauseViewType1, StateChangeCauseViewType2, StateChangeCauseViewType3, StateChangeCauseViewType4]]):
    pass

