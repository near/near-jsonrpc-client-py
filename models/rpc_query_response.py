from models.access_key_list import AccessKeyList
from models.access_key_view import AccessKeyView
from models.account_view import AccountView
from models.call_result import CallResult
from models.contract_code_view import ContractCodeView
from models.crypto_hash import CryptoHash
from models.gas_key_list import GasKeyList
from models.gas_key_view import GasKeyView
from models.view_state_result import ViewStateResult
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
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

class RpcQueryResponseGasKeyView(GasKeyView):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)

class RpcQueryResponseGasKeyList(GasKeyList):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)

class RpcQueryResponse(RootModel[Union[RpcQueryResponseAccountView, RpcQueryResponseContractCodeView, RpcQueryResponseViewStateResult, RpcQueryResponseCallResult, RpcQueryResponseAccessKeyView, RpcQueryResponseAccessKeyList, RpcQueryResponseGasKeyView, RpcQueryResponseGasKeyList]]):
    pass

