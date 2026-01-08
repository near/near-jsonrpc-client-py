from near_jsonrpc_models.delegate_action import DelegateAction
from near_jsonrpc_models.signature import Signature
from pydantic import BaseModel


class SignedDelegateAction(BaseModel):
    delegate_action: DelegateAction
    signature: Signature
