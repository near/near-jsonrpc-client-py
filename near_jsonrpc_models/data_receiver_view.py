from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel


class DataReceiverView(BaseModel):
    data_id: CryptoHash
    receiver_id: AccountId
