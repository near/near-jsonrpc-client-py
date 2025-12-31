from models.account_id import AccountId
from models.block_id import BlockId
from models.finality import Finality
from models.function_args import FunctionArgs
from models.sync_checkpoint import SyncCheckpoint
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class RpcCallFunctionRequestBlockId(BaseModel):
    account_id: AccountId
    args_base64: FunctionArgs
    method_name: str
    block_id: BlockId

class RpcCallFunctionRequestFinality(BaseModel):
    account_id: AccountId
    args_base64: FunctionArgs
    method_name: str
    finality: Finality

class RpcCallFunctionRequestSyncCheckpoint(BaseModel):
    account_id: AccountId
    args_base64: FunctionArgs
    method_name: str
    sync_checkpoint: SyncCheckpoint

class RpcCallFunctionRequest(RootModel[Union[RpcCallFunctionRequestBlockId, RpcCallFunctionRequestFinality, RpcCallFunctionRequestSyncCheckpoint]]):
    pass

