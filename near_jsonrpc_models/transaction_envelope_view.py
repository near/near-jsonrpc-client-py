"""A transaction in a chunk. Untagged so that a signed transaction keeps the JSON shape it
always had."""

from near_jsonrpc_models.ecc_transaction_view import EccTransactionView
from near_jsonrpc_models.signed_transaction_view import SignedTransactionView
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class TransactionEnvelopeViewSignedTransactionView(SignedTransactionView):
    pass

class TransactionEnvelopeViewEccTransactionView(EccTransactionView):
    pass

class TransactionEnvelopeView(RootModel[Union[TransactionEnvelopeViewSignedTransactionView, TransactionEnvelopeViewEccTransactionView]]):
    pass

