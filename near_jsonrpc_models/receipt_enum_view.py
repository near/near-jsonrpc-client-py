from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.action_view import ActionView
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.data_receiver_view import DataReceiverView
from near_jsonrpc_models.global_contract_identifier import GlobalContractIdentifier
from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.shard_id import ShardId
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import List
from typing import Union


class ReceiptEnumViewActionPayload(BaseModel):
    actions: List[ActionView]
    gas_price: NearToken
    input_data_ids: List[CryptoHash]
    is_promise_yield: bool = False
    output_data_receivers: List[DataReceiverView]
    refund_to: AccountId | None = None
    signer_id: AccountId
    signer_public_key: PublicKey

class ReceiptEnumViewAction(StrictBaseModel):
    Action: ReceiptEnumViewActionPayload

class ReceiptEnumViewDataPayload(BaseModel):
    data: str | None = None
    data_id: CryptoHash
    is_promise_resume: bool = False

class ReceiptEnumViewData(StrictBaseModel):
    Data: ReceiptEnumViewDataPayload

class ReceiptEnumViewGlobalContractDistributionPayload(BaseModel):
    already_delivered_shards: List[ShardId]
    code: str
    id: GlobalContractIdentifier
    target_shard: ShardId

class ReceiptEnumViewGlobalContractDistribution(StrictBaseModel):
    GlobalContractDistribution: ReceiptEnumViewGlobalContractDistributionPayload

class ReceiptEnumView(RootModel[Union[ReceiptEnumViewAction, ReceiptEnumViewData, ReceiptEnumViewGlobalContractDistribution]]):
    pass

