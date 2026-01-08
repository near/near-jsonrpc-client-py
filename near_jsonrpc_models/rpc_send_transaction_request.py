from near_jsonrpc_models.signed_transaction import SignedTransaction
from near_jsonrpc_models.tx_execution_status import TxExecutionStatus
from pydantic import BaseModel
from pydantic import Field


class RpcSendTransactionRequest(BaseModel):
    signed_tx_base64: SignedTransaction
    wait_until: TxExecutionStatus = Field(default_factory=lambda: TxExecutionStatus('EXECUTED_OPTIMISTIC'))
