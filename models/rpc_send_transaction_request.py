from models.signed_transaction import SignedTransaction
from models.tx_execution_status import TxExecutionStatus
from pydantic import BaseModel
from pydantic import Field


class RpcSendTransactionRequest(BaseModel):
    signed_tx_base64: SignedTransaction
    wait_until: TxExecutionStatus = Field(default_factory=lambda: TxExecutionStatus('EXECUTED_OPTIMISTIC'))
