from near_jsonrpc_models.signature import Signature
from near_jsonrpc_models.versioned_delegate_action_payload import VersionedDelegateActionPayload
from pydantic import BaseModel


class VersionedSignedDelegateAction(BaseModel):
    delegate_action: VersionedDelegateActionPayload
    signature: Signature
