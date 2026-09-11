from near_jsonrpc_models.execution_outcome_with_id_view import ExecutionOutcomeWithIdView
from near_jsonrpc_models.receipt_view import ReceiptView
from pydantic import BaseModel


class IndexerExecutionOutcomeWithReceipt(BaseModel):
    execution_outcome: ExecutionOutcomeWithIdView
    receipt: ReceiptView
