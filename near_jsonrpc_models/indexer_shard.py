from near_jsonrpc_models.indexer_chunk_view import IndexerChunkView
from near_jsonrpc_models.indexer_execution_outcome_with_receipt import IndexerExecutionOutcomeWithReceipt
from near_jsonrpc_models.shard_id import ShardId
from near_jsonrpc_models.state_change_with_cause_view import StateChangeWithCauseView
from pydantic import BaseModel
from typing import List


class IndexerShard(BaseModel):
    chunk: IndexerChunkView | None = None
    receipt_execution_outcomes: List[IndexerExecutionOutcomeWithReceipt]
    shard_id: ShardId
    state_changes: List[StateChangeWithCauseView]
