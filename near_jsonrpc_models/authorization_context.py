"""What authorized a chain of receipts: an access key that signed a transaction, or an
external call its contract accepted."""

from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class AuthorizationContextExternalContractCall(RootModel[Literal['ExternalContractCall']]):
    pass

class AuthorizationContextAccessKey(StrictBaseModel):
    AccessKey: PublicKey

class AuthorizationContext(RootModel[Union[AuthorizationContextExternalContractCall, AuthorizationContextAccessKey]]):
    pass

