from models.account_id import AccountId
from models.chunk_header_view import ChunkHeaderView
from models.receipt_view import ReceiptView
from models.signed_transaction_view import SignedTransactionView
from pydantic import BaseModel
from typing import List


class RpcChunkResponse(BaseModel):
    author: AccountId
    header: ChunkHeaderView
    receipts: List[ReceiptView]
    transactions: List[SignedTransactionView]
