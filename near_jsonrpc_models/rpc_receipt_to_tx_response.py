from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel


class RpcReceiptToTxResponse(BaseModel):
    sender_account_id: AccountId
    transaction_hash: CryptoHash
