from near_jsonrpc_models.access_key_list import AccessKeyList
from near_jsonrpc_models.access_key_view import AccessKeyView
from near_jsonrpc_models.account_view import AccountView
from near_jsonrpc_models.call_result import CallResult
from near_jsonrpc_models.contract_code_view import ContractCodeView
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.view_state_result import ViewStateResult
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Any
from typing import Union


class RpcQueryResponseAccountView(AccountView):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)

class RpcQueryResponseContractCodeView(ContractCodeView):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)

class RpcQueryResponseViewStateResult(ViewStateResult):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)

class RpcQueryResponseCallResult(CallResult):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)

class RpcQueryResponseAccessKeyView(AccessKeyView):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)

class RpcQueryResponseAccessKeyList(AccessKeyList):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)

class RpcQueryResponse(RootModel[Union[RpcQueryResponseAccountView, RpcQueryResponseContractCodeView, RpcQueryResponseViewStateResult, RpcQueryResponseCallResult, RpcQueryResponseAccessKeyView, RpcQueryResponseAccessKeyList, Any]]):
    pass

