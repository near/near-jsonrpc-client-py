from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.chunk_header_view import ChunkHeaderView
from near_jsonrpc_models.receipt_view import ReceiptView
from near_jsonrpc_models.signed_transaction_view import SignedTransactionView
from pydantic import BaseModel
from typing import List


class RpcChunkResponse(BaseModel):
    author: AccountId
    header: ChunkHeaderView
    receipts: List[ReceiptView]
    transactions: List[SignedTransactionView]
