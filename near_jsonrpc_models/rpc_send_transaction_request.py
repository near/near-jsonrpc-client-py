"""On the wire, a signed transaction goes in `signed_tx_base64` and an ECC in
`external_call_base64`, so requests with a signed transaction look as they did before ECCs."""

from near_jsonrpc_models.ecc_transaction import EccTransaction
from near_jsonrpc_models.signed_transaction import SignedTransaction
from near_jsonrpc_models.tx_execution_status import TxExecutionStatus
from pydantic import BaseModel
from pydantic import Field


class RpcSendTransactionRequest(BaseModel):
    # An external contract call (ECC). Exactly one of this and `signed_tx_base64` is expected.
    external_call_base64: EccTransaction | None = None
    # A signed transaction. Exactly one of this and `external_call_base64` is expected.
    signed_tx_base64: SignedTransaction | None = None
    wait_until: TxExecutionStatus = Field(default_factory=lambda: TxExecutionStatus('EXECUTED_OPTIMISTIC'))
