from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_id import BlockId
from near_jsonrpc_models.finality import Finality
from near_jsonrpc_models.function_args import FunctionArgs
from near_jsonrpc_models.sync_checkpoint import SyncCheckpoint
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

