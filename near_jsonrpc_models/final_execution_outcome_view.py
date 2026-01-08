"""Execution outcome of the transaction and all the subsequent receipts.
Could be not finalized yet"""

from near_jsonrpc_models.execution_outcome_with_id_view import ExecutionOutcomeWithIdView
from near_jsonrpc_models.final_execution_status import FinalExecutionStatus
from near_jsonrpc_models.signed_transaction_view import SignedTransactionView
from pydantic import BaseModel
from typing import List


class FinalExecutionOutcomeView(BaseModel):
    # The execution outcome of receipts.
    receipts_outcome: List[ExecutionOutcomeWithIdView]
    # Execution status defined by chain.rs:get_final_transaction_result
    # FinalExecutionStatus::NotStarted - the tx is not converted to the receipt yet
    # FinalExecutionStatus::Started - we have at least 1 receipt, but the first leaf receipt_id (using dfs) hasn't finished the execution
    # FinalExecutionStatus::Failure - the result of the first leaf receipt_id
    # FinalExecutionStatus::SuccessValue - the result of the first leaf receipt_id
    status: FinalExecutionStatus
    # Signed Transaction
    transaction: SignedTransactionView
    # The execution outcome of the signed transaction.
    transaction_outcome: ExecutionOutcomeWithIdView
