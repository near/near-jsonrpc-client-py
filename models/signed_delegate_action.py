from models.delegate_action import DelegateAction
from models.signature import Signature
from pydantic import BaseModel


class SignedDelegateAction(BaseModel):
    delegate_action: DelegateAction
    signature: Signature
