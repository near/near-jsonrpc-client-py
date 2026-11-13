"""A transaction as a chunk holds it, written like a `TransactionEnvelopeView`."""

from near_jsonrpc_models.chunk_ecc_transaction_view import ChunkEccTransactionView
from near_jsonrpc_models.signed_transaction_view import SignedTransactionView
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class ChunkTransactionViewKind(SignedTransactionView):
    kind: Literal['signed']

class ChunkTransactionViewKind1(ChunkEccTransactionView):
    kind: Literal['external_call']

class ChunkTransactionView(RootModel[Union[ChunkTransactionViewKind, ChunkTransactionViewKind1]]):
    pass

