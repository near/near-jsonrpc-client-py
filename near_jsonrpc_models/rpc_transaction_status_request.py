from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.signed_transaction import SignedTransaction
from near_jsonrpc_models.tx_execution_status import TxExecutionStatus
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcTransactionStatusRequestOption1Option(BaseModel):
    wait_until: TxExecutionStatus = 'EXECUTED_OPTIMISTIC'
    signed_tx_base64: SignedTransaction

class RpcTransactionStatusRequestOption2Option(BaseModel):
    wait_until: TxExecutionStatus = 'EXECUTED_OPTIMISTIC'
    sender_account_id: AccountId
    tx_hash: CryptoHash

class RpcTransactionStatusRequest(RootModel[Union[RpcTransactionStatusRequestOption1Option, RpcTransactionStatusRequestOption2Option]]):
    pass

