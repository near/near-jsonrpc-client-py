"""See crate::types::StateChangeCause for details."""

from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class StateChangeCauseViewNotWritableToDisk(BaseModel):
    type: Literal['not_writable_to_disk']

class StateChangeCauseViewInitialState(BaseModel):
    type: Literal['initial_state']

class StateChangeCauseViewTransactionProcessing(BaseModel):
    tx_hash: CryptoHash
    type: Literal['transaction_processing']

class StateChangeCauseViewActionReceiptProcessingStarted(BaseModel):
    receipt_hash: CryptoHash
    type: Literal['action_receipt_processing_started']

class StateChangeCauseViewActionReceiptGasReward(BaseModel):
    receipt_hash: CryptoHash
    type: Literal['action_receipt_gas_reward']

class StateChangeCauseViewReceiptProcessing(BaseModel):
    receipt_hash: CryptoHash
    type: Literal['receipt_processing']

class StateChangeCauseViewPostponedReceipt(BaseModel):
    receipt_hash: CryptoHash
    type: Literal['postponed_receipt']

class StateChangeCauseViewUpdatedDelayedReceipts(BaseModel):
    type: Literal['updated_delayed_receipts']

class StateChangeCauseViewValidatorAccountsUpdate(BaseModel):
    type: Literal['validator_accounts_update']

class StateChangeCauseViewMigration(BaseModel):
    type: Literal['migration']

class StateChangeCauseViewBandwidthSchedulerStateUpdate(BaseModel):
    type: Literal['bandwidth_scheduler_state_update']

class StateChangeCauseView(RootModel[Union[StateChangeCauseViewNotWritableToDisk, StateChangeCauseViewInitialState, StateChangeCauseViewTransactionProcessing, StateChangeCauseViewActionReceiptProcessingStarted, StateChangeCauseViewActionReceiptGasReward, StateChangeCauseViewReceiptProcessing, StateChangeCauseViewPostponedReceipt, StateChangeCauseViewUpdatedDelayedReceipts, StateChangeCauseViewValidatorAccountsUpdate, StateChangeCauseViewMigration, StateChangeCauseViewBandwidthSchedulerStateUpdate]]):
    pass

