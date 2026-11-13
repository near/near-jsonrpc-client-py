"""A transaction of either kind, in a chunk, a tx status result or an indexer message."""

from near_jsonrpc_models.ecc_transaction_view import EccTransactionView
from near_jsonrpc_models.signed_transaction_view import SignedTransactionView
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class TransactionEnvelopeViewKind(SignedTransactionView):
    kind: Literal['signed']

class TransactionEnvelopeViewKind1(EccTransactionView):
    kind: Literal['external_call']

class TransactionEnvelopeView(RootModel[Union[TransactionEnvelopeViewKind, TransactionEnvelopeViewKind1]]):
    pass

